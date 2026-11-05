// Environment settings. Each is read as `TS_PORTABLE_<NAME>`; when that is unset,
// the spelling used before 1.1.0, `ONNX_TOTALSEG_<NAME>`, is still accepted.

use std::{env, ffi::OsString};

const PREFIX: &str = "TS_PORTABLE_";
const OLD_PREFIX: &str = "ONNX_TOTALSEG_";

/// `env::var` for the setting `name` (without prefix).
pub fn var(name: &str) -> Result<String, env::VarError> {
    match env::var(format!("{PREFIX}{name}")) {
        Err(env::VarError::NotPresent) => env::var(format!("{OLD_PREFIX}{name}")),
        other => other,
    }
}

/// `env::var_os` for the setting `name` (without prefix).
pub fn var_os(name: &str) -> Option<OsString> {
    env::var_os(format!("{PREFIX}{name}")).or_else(|| env::var_os(format!("{OLD_PREFIX}{name}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    // One test, with names no other code reads, so that no other thread touches
    // these variables while they are set.
    #[test]
    fn new_name_wins_and_old_name_still_works() {
        unsafe {
            env::set_var("ONNX_TOTALSEG_SETTINGS_TEST_OLD", "old");
            env::set_var("ONNX_TOTALSEG_SETTINGS_TEST_BOTH", "old");
            env::set_var("TS_PORTABLE_SETTINGS_TEST_BOTH", "new");
        }
        assert_eq!(var("SETTINGS_TEST_OLD").as_deref(), Ok("old"));
        assert_eq!(var("SETTINGS_TEST_BOTH").as_deref(), Ok("new"));
        assert_eq!(var_os("SETTINGS_TEST_OLD"), Some("old".into()));
        assert_eq!(var_os("SETTINGS_TEST_BOTH"), Some("new".into()));
        assert_eq!(var("SETTINGS_TEST_UNSET"), Err(env::VarError::NotPresent));
        assert_eq!(var_os("SETTINGS_TEST_UNSET"), None);
    }
}
