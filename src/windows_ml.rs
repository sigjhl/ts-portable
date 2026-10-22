#[cfg(not(windows))]
use std::path::Path;

#[cfg(not(windows))]
use anyhow::Result;

// OpenVINO is handled as a first-class, self-bundled direct execution provider
// (`--device openvino`, built via `ep::OpenVINO`), not through the Windows-ML
// catalog. This lets it work cross-platform (incl. Linux) with the OpenVINO
// runtime we ship, instead of relying on a Windows-only certified-EP catalog.
pub const CATALOG_PROVIDER_ORDER: &[&str] = &[
    "QNNExecutionProvider",
    "VitisAIExecutionProvider",
    "MIGraphXExecutionProvider",
    "NvTensorRtRtxExecutionProvider",
];

pub fn provider_name_eq(left: &str, right: &str) -> bool {
    normalize_provider_name(left) == normalize_provider_name(right)
}

pub fn is_catalog_provider_name(name: &str) -> bool {
    CATALOG_PROVIDER_ORDER
        .iter()
        .any(|known| provider_name_eq(name, known))
        || provider_name_eq(name, "NvTensorRTRTXExecutionProvider")
}

fn normalize_provider_name(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

#[cfg(not(windows))]
pub fn register_installed_certified(_ort_lib: Option<&Path>) -> Result<Vec<String>> {
    anyhow::bail!(
        "Windows ML execution provider catalog is only available on Windows; use --device directml for legacy DirectML or --device cpu on this platform"
    )
}

#[cfg(not(windows))]
pub fn registered_provider_names() -> Result<Vec<String>> {
    Ok(Vec::new())
}

#[cfg(not(windows))]
pub fn provider_is_registered(_name: &str) -> Result<bool> {
    Ok(false)
}

#[cfg(not(windows))]
pub fn register_byo_provider_libraries() -> Result<Vec<String>> {
    Ok(Vec::new())
}

#[cfg(windows)]
mod imp {
    use std::{
        env,
        ffi::{CStr, c_char, c_void},
        path::{Path, PathBuf},
        ptr,
        sync::OnceLock,
    };

    use anyhow::{Context, Result, anyhow};
    use libloading::Library;
    use ort::environment::Environment;

    use super::{CATALOG_PROVIDER_ORDER, provider_name_eq};

    type HResult = i32;
    type Bool = i32;
    type WinMLEpCatalogHandle = *mut c_void;
    type WinMLEpHandle = *mut c_void;

    const TRUE: Bool = 1;
    const WINML_EP_READY_STATE_NOT_PRESENT: i32 = 2;
    const WINML_EP_CERTIFICATION_CERTIFIED: i32 = 1;

    #[repr(C)]
    struct WinMLEpInfo {
        name: *const c_char,
        version: *const c_char,
        package_family_name: *const c_char,
        library_path: *const c_char,
        package_root_path: *const c_char,
        ready_state: i32,
        certification: i32,
    }

    type WinMLEpEnumCallback =
        Option<unsafe extern "system" fn(WinMLEpHandle, *const WinMLEpInfo, *mut c_void) -> Bool>;
    type WinMLEpCatalogCreate = unsafe extern "system" fn(*mut WinMLEpCatalogHandle) -> HResult;
    type WinMLEpCatalogRelease = unsafe extern "system" fn(WinMLEpCatalogHandle);
    type WinMLEpCatalogEnumProviders = unsafe extern "system" fn(
        WinMLEpCatalogHandle,
        WinMLEpEnumCallback,
        *mut c_void,
    ) -> HResult;
    type WinMLEpEnsureReady = unsafe extern "system" fn(WinMLEpHandle) -> HResult;
    type WinMLEpGetLibraryPathSize =
        unsafe extern "system" fn(WinMLEpHandle, *mut usize) -> HResult;
    type WinMLEpGetLibraryPath =
        unsafe extern "system" fn(WinMLEpHandle, usize, *mut c_char, *mut usize) -> HResult;

    struct CatalogApi {
        _library: Library,
        catalog_create: WinMLEpCatalogCreate,
        catalog_release: WinMLEpCatalogRelease,
        catalog_enum_providers: WinMLEpCatalogEnumProviders,
        ensure_ready: WinMLEpEnsureReady,
        get_library_path_size: WinMLEpGetLibraryPathSize,
        get_library_path: WinMLEpGetLibraryPath,
    }

    struct CatalogGuard<'a> {
        api: &'a CatalogApi,
        catalog: WinMLEpCatalogHandle,
    }

    impl Drop for CatalogGuard<'_> {
        fn drop(&mut self) {
            if !self.catalog.is_null() {
                unsafe { (self.api.catalog_release)(self.catalog) };
            }
        }
    }

    #[derive(Debug)]
    struct ProviderLibrary {
        registration_name: String,
        ep_name: String,
        library_path: PathBuf,
    }

    struct EnumContext<'a> {
        api: &'a CatalogApi,
        providers: Vec<ProviderLibrary>,
        warnings: Vec<String>,
    }

    pub fn register_installed_certified(ort_lib: Option<&Path>) -> Result<Vec<String>> {
        static REGISTERED: OnceLock<Result<Vec<String>, String>> = OnceLock::new();
        REGISTERED
            .get_or_init(|| {
                register_installed_certified_inner(ort_lib).map_err(|err| format!("{err:#}"))
            })
            .clone()
            .map_err(anyhow::Error::msg)
    }

    pub fn registered_provider_names() -> Result<Vec<String>> {
        let env = Environment::current().map_err(|e| anyhow!("{e}"))?;
        let mut names: Vec<String> = Vec::new();
        for device in env.devices() {
            let Ok(name) = device.ep() else {
                continue;
            };
            if !names.iter().any(|known| provider_name_eq(known, name)) {
                names.push(name.to_string());
            }
        }
        Ok(names)
    }

    pub fn provider_is_registered(name: &str) -> Result<bool> {
        Ok(registered_provider_names()?
            .iter()
            .any(|registered| provider_name_eq(registered, name)))
    }

    pub fn register_byo_provider_libraries() -> Result<Vec<String>> {
        static REGISTERED: OnceLock<Result<Vec<String>, String>> = OnceLock::new();
        REGISTERED
            .get_or_init(|| {
                register_byo_provider_libraries_inner().map_err(|err| format!("{err:#}"))
            })
            .clone()
            .map_err(anyhow::Error::msg)
    }

    fn register_byo_provider_libraries_inner() -> Result<Vec<String>> {
        let specs = env::var("ONNX_TOTALSEG_WINDOWS_ML_EP_LIBS")
            .or_else(|_| env::var("ONNX_TOTALSEG_WINDOWSML_EP_LIBS"))
            .unwrap_or_default();
        if specs.trim().is_empty() {
            return Ok(Vec::new());
        }

        let env = Environment::current().map_err(|e| anyhow!("{e}"))?;
        let mut registered = Vec::new();
        for spec in specs
            .split(';')
            .map(str::trim)
            .filter(|spec| !spec.is_empty())
        {
            let (registration_name, path) = spec.split_once('=').ok_or_else(|| {
                anyhow!("invalid Windows ML BYO EP spec '{spec}'; expected registration_name=path")
            })?;
            let registration_name = byo_registration_name(registration_name.trim());
            let path = PathBuf::from(path.trim());
            if registration_name.is_empty() || path.as_os_str().is_empty() {
                anyhow::bail!(
                    "invalid Windows ML BYO EP spec '{spec}'; expected non-empty registration_name=path"
                );
            }
            env.register_ep_library(&registration_name, &path)
                .map_err(|e| anyhow!("{e}"))
                .with_context(|| {
                    format!(
                        "failed to register Windows ML BYO EP {registration_name} from {}",
                        path.display()
                    )
                })?;
            registered.push(registration_name);
        }
        Ok(registered)
    }

    fn byo_registration_name(name: &str) -> String {
        CATALOG_PROVIDER_ORDER
            .iter()
            .copied()
            .find(|ep_name| {
                provider_name_eq(name, ep_name)
                    || provider_name_eq(name, ep_name.trim_end_matches("ExecutionProvider"))
            })
            .unwrap_or(name)
            .to_string()
    }

    fn register_installed_certified_inner(ort_lib: Option<&Path>) -> Result<Vec<String>> {
        let api = load_catalog_api(ort_lib)?;
        let env = Environment::current().map_err(|e| anyhow!("{e}"))?;
        let mut catalog = ptr::null_mut();
        check_hr(
            unsafe { (api.catalog_create)(&mut catalog) },
            "WinMLEpCatalogCreate",
        )?;
        let _catalog = CatalogGuard { api: &api, catalog };

        let mut ctx = EnumContext {
            api: &api,
            providers: Vec::new(),
            warnings: Vec::new(),
        };
        check_hr(
            unsafe {
                (api.catalog_enum_providers)(
                    catalog,
                    Some(enum_installed_certified_provider),
                    (&mut ctx as *mut EnumContext<'_>).cast::<c_void>(),
                )
            },
            "WinMLEpCatalogEnumProviders",
        )?;

        let mut registered: Vec<String> = Vec::new();
        for provider in ctx.providers {
            match env.register_ep_library(&provider.registration_name, &provider.library_path) {
                Ok(_) => {
                    if !registered
                        .iter()
                        .any(|name| provider_name_eq(name, &provider.ep_name))
                    {
                        registered.push(provider.ep_name);
                    }
                }
                Err(err) => ctx.warnings.push(format!(
                    "failed to register Windows ML EP {} from {}: {err}",
                    provider.registration_name,
                    provider.library_path.display()
                )),
            }
        }

        for warning in ctx.warnings {
            eprintln!("[windowsml] {warning}");
        }

        Ok(registered)
    }

    unsafe extern "system" fn enum_installed_certified_provider(
        ep: WinMLEpHandle,
        info: *const WinMLEpInfo,
        context: *mut c_void,
    ) -> Bool {
        let ctx = unsafe { &mut *(context.cast::<EnumContext<'_>>()) };
        if info.is_null() {
            return TRUE;
        }
        let info = unsafe { &*info };
        if info.name.is_null() || info.certification != WINML_EP_CERTIFICATION_CERTIFIED {
            return TRUE;
        }
        if info.ready_state == WINML_EP_READY_STATE_NOT_PRESENT && !install_missing_providers() {
            return TRUE;
        }

        let registration_name = unsafe { CStr::from_ptr(info.name) }
            .to_string_lossy()
            .into_owned();
        let Some(ep_name) = ep_name_from_registration_name(&registration_name) else {
            return TRUE;
        };

        let hr = unsafe { (ctx.api.ensure_ready)(ep) };
        if failed(hr) {
            ctx.warnings.push(format!(
                "skipping Windows ML EP {registration_name}: WinMLEpEnsureReady failed with HRESULT 0x{hr:08X}"
            ));
            return TRUE;
        }

        match get_library_path(ctx.api, ep) {
            Ok(library_path) => ctx.providers.push(ProviderLibrary {
                registration_name,
                ep_name: ep_name.to_string(),
                library_path,
            }),
            Err(err) => ctx.warnings.push(format!(
                "skipping Windows ML EP {registration_name}: {err:#}"
            )),
        }
        TRUE
    }

    fn ep_name_from_registration_name(name: &str) -> Option<&'static str> {
        CATALOG_PROVIDER_ORDER
            .iter()
            .copied()
            .find(|ep_name| provider_name_eq(name, ep_name.trim_end_matches("ExecutionProvider")))
            .or_else(|| {
                CATALOG_PROVIDER_ORDER
                    .iter()
                    .copied()
                    .find(|ep_name| provider_name_eq(name, ep_name))
            })
    }

    fn install_missing_providers() -> bool {
        env::var("ONNX_TOTALSEG_WINDOWS_ML_INSTALL")
            .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    }

    fn get_library_path(api: &CatalogApi, ep: WinMLEpHandle) -> Result<PathBuf> {
        let mut size = 0usize;
        check_hr(
            unsafe { (api.get_library_path_size)(ep, &mut size) },
            "WinMLEpGetLibraryPathSize",
        )?;
        anyhow::ensure!(size > 0, "Windows ML EP reported an empty library path");

        let mut bytes = vec![0u8; size];
        let mut used = 0usize;
        check_hr(
            unsafe {
                (api.get_library_path)(ep, size, bytes.as_mut_ptr().cast::<c_char>(), &mut used)
            },
            "WinMLEpGetLibraryPath",
        )?;
        let end = bytes
            .iter()
            .position(|b| *b == 0)
            .unwrap_or_else(|| used.min(bytes.len()));
        let path = String::from_utf8(bytes[..end].to_vec())
            .context("Windows ML EP library path was not valid UTF-8")?;
        Ok(PathBuf::from(path))
    }

    fn load_catalog_api(ort_lib: Option<&Path>) -> Result<CatalogApi> {
        let mut errors = Vec::new();
        for candidate in library_candidates(ort_lib) {
            let library = match unsafe { Library::new(&candidate) } {
                Ok(library) => library,
                Err(err) => {
                    errors.push(format!("{}: {err}", candidate.display()));
                    continue;
                }
            };
            return unsafe { api_from_library(library) }
                .with_context(|| format!("failed to load symbols from {}", candidate.display()));
        }
        anyhow::bail!(
            "failed to load Microsoft.Windows.AI.MachineLearning.dll; tried: {}",
            errors.join("; ")
        )
    }

    unsafe fn api_from_library(library: Library) -> Result<CatalogApi> {
        let catalog_create = unsafe { load_symbol(&library, b"WinMLEpCatalogCreate\0")? };
        let catalog_release = unsafe { load_symbol(&library, b"WinMLEpCatalogRelease\0")? };
        let catalog_enum_providers =
            unsafe { load_symbol(&library, b"WinMLEpCatalogEnumProviders\0")? };
        let ensure_ready = unsafe { load_symbol(&library, b"WinMLEpEnsureReady\0")? };
        let get_library_path_size =
            unsafe { load_symbol(&library, b"WinMLEpGetLibraryPathSize\0")? };
        let get_library_path = unsafe { load_symbol(&library, b"WinMLEpGetLibraryPath\0")? };
        Ok(CatalogApi {
            _library: library,
            catalog_create,
            catalog_release,
            catalog_enum_providers,
            ensure_ready,
            get_library_path_size,
            get_library_path,
        })
    }

    unsafe fn load_symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T> {
        Ok(*unsafe { library.get::<T>(name)? })
    }

    fn library_candidates(ort_lib: Option<&Path>) -> Vec<PathBuf> {
        let mut out = Vec::new();
        if let Some(path) = env::var_os("ONNX_TOTALSEG_WINDOWS_ML_LIB")
            .or_else(|| env::var_os("ONNX_TOTALSEG_WINDOWSML_LIB"))
        {
            out.push(PathBuf::from(path));
        }
        if let Some(parent) = ort_lib.and_then(Path::parent) {
            out.push(parent.join("Microsoft.Windows.AI.MachineLearning.dll"));
        }
        if let Ok(exe) = env::current_exe() {
            if let Some(parent) = exe.parent() {
                out.push(parent.join("Microsoft.Windows.AI.MachineLearning.dll"));
            }
        }
        out.push(PathBuf::from("Microsoft.Windows.AI.MachineLearning.dll"));
        out
    }

    fn check_hr(hr: HResult, what: &str) -> Result<()> {
        if failed(hr) {
            anyhow::bail!("{what} failed with HRESULT 0x{hr:08X}");
        }
        Ok(())
    }

    fn failed(hr: HResult) -> bool {
        hr < 0
    }
}

#[cfg(windows)]
pub use imp::{
    provider_is_registered, register_byo_provider_libraries, register_installed_certified,
    registered_provider_names,
};
