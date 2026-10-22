use std::{
    cell::Cell,
    collections::HashMap,
    env, fs,
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex, OnceLock},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use ndarray::ArrayViewD;
use ort::{
    environment::Environment,
    ep::{
        self, ArbitrarilyConfigurableExecutionProvider, ArenaExtendStrategy, ExecutionProvider,
        cuda::ConvAlgorithmSearch,
    },
    memory::Allocator,
    session::{
        HasSelectedOutputs, OutputSelector, RunOptions, Session, builder::GraphOptimizationLevel,
    },
    value::{Tensor, TensorRef},
};
use rayon::prelude::*;

use crate::{
    io::{Affine, ImageProps, lowres_affine},
    resample::{
        change_spacing_to_shape, change_spacing_to_spacing, compute_new_shape,
        resample_data_or_seg_to_shape, resize_labels_to_shape,
    },
    runtime::model::ModelCard,
    volume::Volume,
    windows_ml,
};

struct StageTimer {
    enabled: bool,
    last: Cell<Instant>,
}

impl StageTimer {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            last: Cell::new(Instant::now()),
        }
    }

    fn lap(&self, name: &str) {
        if !self.enabled {
            return;
        }
        let now = Instant::now();
        let dt = now.duration_since(self.last.get());
        self.last.set(now);
        eprintln!("[profile] {name}: {:.3}s", dt.as_secs_f64());
    }
}

#[derive(Clone, Debug)]
pub struct RuntimeOptions {
    pub providers: Option<Vec<String>>,
    pub device: String,
    pub threads: Option<usize>,
    pub fp16: bool,
    pub int8: bool,
    pub ort_lib: Option<PathBuf>,
    pub profile: bool,
    pub low_memory_cpu: bool,
    pub cuda_low_memory: bool,
    pub lazy_session: bool,
}

pub struct Segmentator {
    card: ModelCard,
    session: Option<Session>,
    onnx: PathBuf,
    onnx_files: Vec<PathBuf>,
    opts: RuntimeOptions,
    provider_names: Vec<String>,
    input_name: String,
    output_name: String,
    run_options: Option<RunOptions<HasSelectedOutputs>>,
    image_resample_order: usize,
    pub active_provider: String,
    profile: bool,
}

enum SlidingWindowOutput {
    Logits(Vec<f32>),
    Labels(Volume<u8>),
}

pub type PredictionOutput = (Volume<u8>, Option<Affine>);

impl Segmentator {
    pub fn new(
        card_path: &Path,
        opts: &RuntimeOptions,
        resample_order: Option<usize>,
    ) -> Result<Self> {
        let t = StageTimer::new(opts.profile);
        ensure_ort(opts)?;
        t.lap("session ensure_ort");
        let card = ModelCard::read(card_path)?;
        t.lap("session read_card");
        let root = card_path
            .parent()
            .context("model card has no parent directory")?
            .to_path_buf();
        let provider_names = resolve_provider_names(
            opts.providers.as_deref(),
            &opts.device,
            opts.ort_lib.clone(),
        )?;
        t.lap("session resolve_providers");
        let active_provider = provider_names
            .first()
            .cloned()
            .unwrap_or_else(|| "CPUExecutionProvider".to_string());
        // INT8 variant selection depends on the active provider (OpenVINO-only),
        // so provider resolution must precede model-file resolution.
        let provider_is_openvino = active_provider == "OpenVINOExecutionProvider";
        let onnx_names = card
            .onnx_files
            .clone()
            .filter(|files| !files.is_empty())
            .unwrap_or_else(|| vec![card.onnx_file.clone()]);
        let onnx_files = onnx_names
            .iter()
            .map(|name| {
                resolve_precision_model_path(
                    &root.join(name),
                    opts.fp16,
                    opts.int8,
                    provider_is_openvino,
                )
            })
            .collect::<Vec<_>>();
        let onnx = onnx_files[0].clone();
        let mut out = Self {
            image_resample_order: resample_order.unwrap_or(card.resampling_data_kwargs.order),
            card,
            session: None,
            onnx,
            onnx_files,
            opts: opts.clone(),
            provider_names,
            input_name: String::new(),
            output_name: String::new(),
            run_options: None,
            active_provider,
            profile: opts.profile,
        };
        if !opts.low_memory_cpu && !opts.lazy_session {
            out.ensure_session()?;
        }
        Ok(out)
    }

    fn ensure_session(&mut self) -> Result<()> {
        if self.session.is_some() {
            return Ok(());
        }
        let t = StageTimer::new(self.profile);
        let opt_level = if self.opts.low_memory_cpu
            && !matches!(
                self.card.name.as_str(),
                "Dataset008_HepaticVessel" | "Dataset952_abdominal_muscles_167subj"
            ) {
            GraphOptimizationLevel::Level2
        } else {
            GraphOptimizationLevel::All
        };
        let mut builder = Session::builder()?
            .with_optimization_level(opt_level)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let task_cpu_model = matches!(
            self.card.name.as_str(),
            "Dataset008_HepaticVessel" | "Dataset952_abdominal_muscles_167subj"
        );
        if self.opts.low_memory_cpu {
            // Fastest CPU is session/workspace dominated; avoid extra packed
            // weights while keeping the default provider path for larger tiers.
            if !task_cpu_model {
                builder = builder
                    .with_prepacking(false)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            }
            builder = builder
                .with_device_allocated_initializers()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        if let Some(t) = self.opts.threads {
            builder = builder
                .with_intra_threads(t)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        t.lap("session build_options");
        let providers = execution_providers_from_names(
            &self.provider_names,
            self.opts.low_memory_cpu,
            self.opts.cuda_low_memory,
        )?;
        if !providers.is_empty() {
            builder = builder
                .with_execution_providers(providers)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        builder = attach_windows_ml_devices(builder, &self.provider_names)?;
        t.lap("session attach_providers");
        let session = builder.commit_from_file(&self.onnx).with_context(|| {
            format!(
                "failed to create ONNX Runtime session for {}",
                self.onnx.display()
            )
        })?;
        t.lap("session commit_from_file");
        self.input_name = session
            .inputs()
            .first()
            .context("model has no inputs")?
            .name()
            .to_string();
        self.output_name = session
            .outputs()
            .first()
            .context("model has no outputs")?
            .name()
            .to_string();
        t.lap("session inspect_inputs");
        self.session = Some(session);
        Ok(())
    }

    fn select_onnx(&mut self, path: &Path) {
        if self.onnx == path {
            return;
        }
        self.session = None;
        self.run_options = None;
        self.input_name.clear();
        self.output_name.clear();
        self.onnx = path.to_path_buf();
    }

    pub fn target_spacing(&self) -> [f32; 3] {
        self.card.target_spacing
    }

    fn quantize_outer_resample(&self) -> bool {
        self.card.quantize_outer_resample.unwrap_or(!matches!(
            self.card.name.as_str(),
            "Dataset008_HepaticVessel"
        ))
    }

    fn uses_outer_resample(&self) -> bool {
        self.card.outer_resample.unwrap_or(!matches!(
            self.card.name.as_str(),
            "Dataset008_HepaticVessel"
        ))
    }

    pub fn labels(&self) -> HashMap<u8, String> {
        self.card.id_to_name()
    }

    pub fn preserves_input_orientation(&self) -> bool {
        self.card.preserve_input_orientation.unwrap_or(false)
    }

    pub fn predict_loaded(
        &mut self,
        data: &Volume<f32>,
        props: &ImageProps,
        lowres: bool,
    ) -> Result<PredictionOutput> {
        let t = StageTimer::new(self.profile);
        let (seg_t, spacing_t) = if self.uses_outer_resample() {
            let mut img_t = change_spacing_to_spacing(
                data,
                props.spacing_zyx,
                self.card.target_spacing,
                self.image_resample_order,
            );
            if self.quantize_outer_resample() {
                img_t = img_t.map(|v| (*v as i32) as f32);
            }
            t.lap("single outer_resample_to_model");
            let seg = self.predict_array(&img_t, self.card.target_spacing)?;
            (seg, self.card.target_spacing)
        } else {
            t.lap("single outer_resample_to_model");
            let seg = self.predict_array(data, props.spacing_zyx)?;
            (seg, props.spacing_zyx)
        };
        t.lap("single predict_array");
        if lowres {
            t.lap("single lowres_affine");
            return Ok((seg_t, Some(lowres_affine(props, spacing_t))));
        }
        let seg = change_spacing_to_shape(&seg_t, data.shape, 0);
        t.lap("single upsample_to_original");
        Ok((seg, None))
    }

    pub fn predict_loaded_owned(
        &mut self,
        data: Volume<f32>,
        props: &ImageProps,
        lowres: bool,
    ) -> Result<PredictionOutput> {
        let t = StageTimer::new(self.profile);
        let original_shape = data.shape;
        let original_spacing = props.spacing_zyx;
        let (seg_t, spacing_t) = if self.uses_outer_resample() {
            let mut img_t = change_spacing_to_spacing(
                &data,
                props.spacing_zyx,
                self.card.target_spacing,
                self.image_resample_order,
            );
            if self.quantize_outer_resample() {
                img_t = img_t.map(|v| (*v as i32) as f32);
            }
            drop(data);
            t.lap("single outer_resample_to_model");
            let seg = self.predict_array(&img_t, self.card.target_spacing)?;
            (seg, self.card.target_spacing)
        } else {
            t.lap("single outer_resample_to_model");
            let seg = self.predict_array(&data, original_spacing)?;
            drop(data);
            (seg, original_spacing)
        };
        t.lap("single predict_array");
        if lowres {
            t.lap("single lowres_affine");
            return Ok((seg_t, Some(lowres_affine(props, spacing_t))));
        }
        let seg = change_spacing_to_shape(&seg_t, original_shape, 0);
        t.lap("single upsample_to_original");
        Ok((seg, None))
    }

    pub fn predict_array(&mut self, data: &Volume<f32>, spacing: [f32; 3]) -> Result<Volume<u8>> {
        let t = StageTimer::new(self.profile);
        let shape_before_crop = data.shape;
        let (cropped, bbox) = crop_to_nonzero(data)?;
        t.lap("core crop_to_nonzero");
        let seg_crop = self.predict_cropped_array(cropped, spacing)?;
        if bbox == full_bbox(shape_before_crop) {
            t.lap("core paste_crop");
            return Ok(seg_crop);
        }
        let mut full = Volume::<u8>::zeros(shape_before_crop);
        full.paste(&seg_crop, &bbox);
        t.lap("core paste_crop");
        Ok(full)
    }

    pub fn predict_cropped_array(
        &mut self,
        cropped: Volume<f32>,
        spacing: [f32; 3],
    ) -> Result<Volume<u8>> {
        let t = StageTimer::new(self.profile);
        let shape_after_crop = cropped.shape;
        let resampled = self.preprocess_cropped_array(cropped, spacing);
        let resize_logits_before_argmax =
            !self.uses_outer_resample() && resampled.shape != shape_after_crop;
        let sw = self.sliding_window_ensemble(&resampled, resize_logits_before_argmax)?;
        t.lap("core sliding_window_ort");
        let seg_net = match sw {
            SlidingWindowOutput::Logits(logits) => {
                if resize_logits_before_argmax {
                    let logits = resize_logits_to_shape(
                        &logits,
                        self.card.num_seg_heads,
                        resampled.shape,
                        shape_after_crop,
                        self.card.target_spacing,
                        spacing,
                    )?;
                    argmax_logits(&logits, self.card.num_seg_heads, shape_after_crop)
                } else {
                    argmax_logits(&logits, self.card.num_seg_heads, resampled.shape)
                }
            }
            SlidingWindowOutput::Labels(labels) => labels,
        };
        t.lap("core argmax");
        let seg_crop = if resize_logits_before_argmax || seg_net.shape == shape_after_crop {
            seg_net
        } else {
            resize_labels_to_shape(&seg_net, shape_after_crop)
        };
        t.lap("core resize_labels_to_crop");
        Ok(seg_crop)
    }

    fn preprocess_cropped_array(&self, mut cropped: Volume<f32>, spacing: [f32; 3]) -> Volume<f32> {
        let t = StageTimer::new(self.profile);
        let normalize_before_resample = self.card.normalize_before_resample.unwrap_or(true);
        if normalize_before_resample {
            ct_normalize(&mut cropped, &self.card);
            t.lap("core normalize");
        }
        let new_shape = compute_new_shape(cropped.shape, spacing, self.card.target_spacing);
        let mut resampled = if new_shape == cropped.shape && spacing == self.card.target_spacing {
            cropped
        } else {
            resample_data_or_seg_to_shape(
                &cropped,
                new_shape,
                spacing,
                self.card.target_spacing,
                self.card.resampling_data_kwargs.order,
                self.card.resampling_data_kwargs.order_z,
                self.card.resampling_data_kwargs.force_separate_z,
            )
        };
        t.lap("core resample_to_net");
        if !normalize_before_resample {
            ct_normalize(&mut resampled, &self.card);
            t.lap("core normalize");
        }
        resampled
    }

    fn run_patch_mirrored_scores(
        &mut self,
        patch_data: &[f32],
        patch: [usize; 3],
        heads: usize,
        combinations: &[[bool; 3]],
        preallocated_output: bool,
    ) -> Result<Vec<f32>> {
        let patch_n = patch[0] * patch[1] * patch[2];
        let input_shape = [1_usize, 1, patch[0], patch[1], patch[2]];
        let mut mirrored = vec![0.0_f32; patch_n];
        let mut scores = vec![0.0_f32; heads * patch_n];
        for &flips in combinations {
            let input_data = if flips.iter().any(|flip| *flip) {
                mirror_patch_input(patch_data, &mut mirrored, patch, flips);
                mirrored.as_slice()
            } else {
                patch_data
            };
            let input = TensorRef::from_array_view((input_shape, input_data))?;
            let outputs = if preallocated_output {
                let run_options = self
                    .run_options
                    .as_ref()
                    .context("multi-patch run options were not initialized")?;
                self.session
                    .as_mut()
                    .expect("session was initialized")
                    .run_with_options(
                        ort::inputs![self.input_name.as_str() => input],
                        run_options,
                    )?
            } else {
                self.session
                    .as_mut()
                    .expect("session was initialized")
                    .run(ort::inputs![self.input_name.as_str() => input])?
            };
            let pred = outputs[0].try_extract_array::<f32>()?;
            let pred_shape = pred.shape();
            if pred_shape.len() != 5 || pred_shape[1] != heads {
                bail!("unexpected logits shape from model: {:?}", pred_shape);
            }
            if let Some(slice) = pred.as_slice_memory_order() {
                accumulate_unmirrored_scores(&mut scores, slice, heads, patch, flips);
            } else {
                accumulate_unmirrored_scores_view(&mut scores, &pred, heads, patch, flips);
            }
        }
        Ok(scores)
    }

    fn sliding_window(
        &mut self,
        data: &Volume<f32>,
        return_logits_for_single_patch: bool,
    ) -> Result<SlidingWindowOutput> {
        self.ensure_session()?;
        let patch = self.card.patch_size;
        let padded_owned;
        let (padded, revert) = if data
            .shape
            .iter()
            .zip(patch)
            .all(|(dim, patch)| *dim >= patch)
        {
            (data, full_bbox(data.shape))
        } else {
            let (owned, revert) = pad_to_patch(data, patch);
            padded_owned = owned;
            (&padded_owned, revert)
        };
        let steps = compute_steps_for_sliding_window(
            padded.shape,
            patch,
            self.card.inference.tile_step_size,
        )?;
        let heads = self.card.num_seg_heads;
        let patch_n = patch[0] * patch[1] * patch[2];
        let mut patch_data = vec![0.0_f32; patch_n];
        let input_shape = [1_usize, 1, patch[0], patch[1], patch[2]];
        let mut profile_patches = 0usize;
        let mut profile_copy_s = 0.0_f64;
        let mut profile_run_s = 0.0_f64;
        let mut profile_accumulate_s = 0.0_f64;
        let mirror_combinations = inference_mirror_combinations(&self.card)?;
        let use_mirroring = mirror_combinations.len() > 1;
        let single_patch = steps.iter().all(|axis| axis.len() == 1 && axis[0] == 0);
        if single_patch {
            profile_patches += 1;
            let profile_t = self.profile.then(Instant::now);
            copy_patch_rows(padded, [0, 0, 0], patch, &mut patch_data);
            if let Some(t) = profile_t {
                profile_copy_s += t.elapsed().as_secs_f64();
            }
            let profile_t = self.profile.then(Instant::now);
            let mirrored_scores = use_mirroring
                .then(|| {
                    self.run_patch_mirrored_scores(
                        &patch_data,
                        patch,
                        heads,
                        &mirror_combinations,
                        false,
                    )
                })
                .transpose()?;
            let outputs = if use_mirroring {
                None
            } else {
                let input = TensorRef::from_array_view((input_shape, patch_data.as_slice()))?;
                Some(
                    self.session
                        .as_mut()
                        .expect("session was initialized")
                        .run(ort::inputs![self.input_name.as_str() => input])?,
                )
            };
            if let Some(t) = profile_t {
                profile_run_s += t.elapsed().as_secs_f64();
            }
            let pred = outputs
                .as_ref()
                .map(|outputs| outputs[0].try_extract_array::<f32>())
                .transpose()?;
            if let Some(pred) = &pred {
                let pred_shape = pred.shape();
                if pred_shape.len() != 5 || pred_shape[1] != heads {
                    bail!("unexpected logits shape from model: {:?}", pred_shape);
                }
            }
            let profile_t = self.profile.then(Instant::now);
            if return_logits_for_single_patch {
                let logits = if let Some(scores) = mirrored_scores {
                    scores
                } else if let Some(pred_slice) = pred
                    .as_ref()
                    .expect("non-mirrored prediction exists")
                    .as_slice_memory_order()
                {
                    pred_slice.to_vec()
                } else {
                    pred.as_ref()
                        .expect("non-mirrored prediction exists")
                        .iter()
                        .copied()
                        .collect()
                };
                if padded.shape == data.shape && revert == full_bbox(data.shape) {
                    return Ok(SlidingWindowOutput::Logits(logits));
                }
                return Ok(SlidingWindowOutput::Logits(unpad_logits(
                    &logits,
                    heads,
                    padded.shape,
                    data.shape,
                    revert,
                )));
            }
            let labels = if let Some(scores) = mirrored_scores {
                argmax_single_patch_logits(&scores, heads, patch, data.shape, revert)
            } else if let Some(pred_slice) = pred
                .as_ref()
                .expect("non-mirrored prediction exists")
                .as_slice_memory_order()
            {
                argmax_single_patch_logits(pred_slice, heads, patch, data.shape, revert)
            } else {
                argmax_single_patch_logits_view(
                    pred.as_ref().expect("non-mirrored prediction exists"),
                    heads,
                    patch,
                    data.shape,
                    revert,
                )
            };
            if let Some(t) = profile_t {
                profile_accumulate_s += t.elapsed().as_secs_f64();
            }
            if self.profile {
                eprintln!(
                    "[profile] core sliding_window_parts: patches={} copy={:.3}s run_extract={:.3}s accumulate={:.3}s",
                    profile_patches, profile_copy_s, profile_run_s, profile_accumulate_s
                );
            }
            return Ok(SlidingWindowOutput::Labels(labels));
        }

        let gaussian = if self.card.inference.use_gaussian {
            Some(cached_gaussian(
                patch,
                self.card.inference.gaussian_sigma_scale,
            ))
        } else {
            None
        };
        if self.run_options.is_none() {
            let output = Tensor::<f32>::new(
                &Allocator::default(),
                [1_usize, heads, patch[0], patch[1], patch[2]],
            )
            .map_err(|e| anyhow::anyhow!("{e}"))?;
            self.run_options = Some(
                RunOptions::new()
                    .map_err(|e| anyhow::anyhow!("{e}"))?
                    .with_outputs(
                        OutputSelector::no_default()
                            .with(self.output_name.clone())
                            .preallocate(self.output_name.clone(), output),
                    ),
            );
        }
        let spatial = padded.shape[0] * padded.shape[1] * padded.shape[2];
        let mut logits = vec![0.0_f32; heads * spatial];
        let mut weight_sum = return_logits_for_single_patch.then(|| vec![0.0_f32; spatial]);
        for &z0 in &steps[0] {
            for &y0 in &steps[1] {
                for &x0 in &steps[2] {
                    profile_patches += 1;
                    let profile_t = self.profile.then(Instant::now);
                    copy_patch_rows(padded, [z0, y0, x0], patch, &mut patch_data);
                    if let Some(t) = profile_t {
                        profile_copy_s += t.elapsed().as_secs_f64();
                    }
                    let profile_t = self.profile.then(Instant::now);
                    let mirrored_scores = use_mirroring
                        .then(|| {
                            self.run_patch_mirrored_scores(
                                &patch_data,
                                patch,
                                heads,
                                &mirror_combinations,
                                true,
                            )
                        })
                        .transpose()?;
                    let outputs = if use_mirroring {
                        None
                    } else {
                        let input =
                            TensorRef::from_array_view((input_shape, patch_data.as_slice()))?;
                        let run_options = self
                            .run_options
                            .as_ref()
                            .expect("multi-patch run options were initialized");
                        Some(
                            self.session
                                .as_mut()
                                .expect("session was initialized")
                                .run_with_options(
                                    ort::inputs![self.input_name.as_str() => input],
                                    run_options,
                                )?,
                        )
                    };
                    if let Some(t) = profile_t {
                        profile_run_s += t.elapsed().as_secs_f64();
                    }
                    let pred = outputs
                        .as_ref()
                        .map(|outputs| outputs[0].try_extract_array::<f32>())
                        .transpose()?;
                    if let Some(pred) = &pred {
                        let pred_shape = pred.shape();
                        if pred_shape.len() != 5 || pred_shape[1] != heads {
                            bail!("unexpected logits shape from model: {:?}", pred_shape);
                        }
                    }
                    let profile_t = self.profile.then(Instant::now);
                    if let Some(scores) = mirrored_scores {
                        accumulate_patch_logits(
                            &mut logits,
                            &scores,
                            gaussian.as_deref(),
                            heads,
                            spatial,
                            padded.shape,
                            patch,
                            [z0, y0, x0],
                        );
                    } else if let Some(pred_slice) = pred
                        .as_ref()
                        .expect("non-mirrored prediction exists")
                        .as_slice_memory_order()
                    {
                        accumulate_patch_logits(
                            &mut logits,
                            pred_slice,
                            gaussian.as_deref(),
                            heads,
                            spatial,
                            padded.shape,
                            patch,
                            [z0, y0, x0],
                        );
                    } else {
                        accumulate_patch_logits_view(
                            &mut logits,
                            pred.as_ref().expect("non-mirrored prediction exists"),
                            gaussian.as_deref(),
                            heads,
                            spatial,
                            padded.shape,
                            patch,
                            [z0, y0, x0],
                        );
                    }
                    if let Some(weight_sum) = &mut weight_sum {
                        accumulate_patch_weights(
                            weight_sum,
                            gaussian.as_deref(),
                            padded.shape,
                            patch,
                            [z0, y0, x0],
                        );
                    }
                    if let Some(t) = profile_t {
                        profile_accumulate_s += t.elapsed().as_secs_f64();
                    }
                }
            }
        }
        if let Some(weight_sum) = &weight_sum {
            normalize_logits_by_weights(&mut logits, weight_sum, heads, spatial);
        }
        if self.profile {
            eprintln!(
                "[profile] core sliding_window_parts: patches={} copy={:.3}s run_extract={:.3}s accumulate={:.3}s",
                profile_patches, profile_copy_s, profile_run_s, profile_accumulate_s
            );
        }
        if padded.shape == data.shape && revert == full_bbox(data.shape) {
            Ok(SlidingWindowOutput::Logits(logits))
        } else {
            Ok(SlidingWindowOutput::Logits(unpad_logits(
                &logits,
                heads,
                padded.shape,
                data.shape,
                revert,
            )))
        }
    }

    fn sliding_window_ensemble(
        &mut self,
        data: &Volume<f32>,
        return_logits_for_single_patch: bool,
    ) -> Result<SlidingWindowOutput> {
        if self.onnx_files.len() == 1 {
            return self.sliding_window(data, return_logits_for_single_patch);
        }
        let files = self.onnx_files.clone();
        let mut ensemble: Option<Vec<f32>> = None;
        for (index, path) in files.iter().enumerate() {
            if self.profile {
                eprintln!(
                    "[profile] ensemble member {}/{}: {}",
                    index + 1,
                    files.len(),
                    path.display()
                );
            }
            self.select_onnx(path);
            let scores = match self.sliding_window(data, true)? {
                SlidingWindowOutput::Logits(scores) => scores,
                SlidingWindowOutput::Labels(_) => {
                    bail!("ensemble member unexpectedly returned labels instead of scores")
                }
            };
            match &mut ensemble {
                Some(sum) => sum
                    .par_iter_mut()
                    .zip(scores.par_iter())
                    .for_each(|(dst, src)| *dst += *src),
                None => ensemble = Some(scores),
            }
        }
        Ok(SlidingWindowOutput::Logits(
            ensemble.context("ensemble requires at least one ONNX model")?,
        ))
    }
}

fn resolve_precision_model_path(
    onnx: &Path,
    fp16_requested: bool,
    int8_requested: bool,
    provider_is_openvino: bool,
) -> PathBuf {
    let variant = |suffix: &str| {
        onnx.with_file_name(format!(
            "{}_{suffix}.onnx",
            onnx.file_stem().unwrap().to_string_lossy()
        ))
    };
    // INT8 is CPU-only and only accelerates under the OpenVINO EP (oneDNN int8
    // kernels lower the QDQ graph); it is a no-op or slower elsewhere, so gate it
    // to that provider. INT8 takes precedence over fp16 when both are set.
    if int8_requested {
        if provider_is_openvino {
            let int8 = variant("int8");
            if int8.exists() {
                return int8;
            }
            eprintln!("[int8] variant {} not found; using fp32", int8.display());
        } else {
            eprintln!(
                "[int8] ignored: requires --device openvino (int8 kernels are OpenVINO-only)"
            );
        }
    }
    if fp16_requested {
        let fp16 = variant("fp16");
        if fp16.exists() {
            return fp16;
        }
        eprintln!("[fp16] variant {} not found; using fp32", fp16.display());
    }
    onnx.to_path_buf()
}

pub struct MultiModelSegmentator {
    card_paths: Vec<PathBuf>,
    parts: Vec<Segmentator>,
    label_map: HashMap<String, u8>,
    opts: RuntimeOptions,
    target_spacing: [f32; 3],
    data_order: usize,
    pub active_provider: String,
    profile: bool,
}

impl MultiModelSegmentator {
    pub fn new(
        card_paths: Vec<PathBuf>,
        label_map_path: &Path,
        opts: RuntimeOptions,
    ) -> Result<Self> {
        let label_text = std::fs::read_to_string(label_map_path)
            .with_context(|| format!("failed to read {}", label_map_path.display()))?;
        let label_map: HashMap<String, u8> = serde_json::from_str(&label_text)?;
        let card0 = ModelCard::read(
            card_paths
                .first()
                .context("full model requires at least one card")?,
        )?;
        let provider_names = resolve_provider_names(
            opts.providers.as_deref(),
            &opts.device,
            opts.ort_lib.clone(),
        )?;
        let active_provider = provider_names
            .first()
            .cloned()
            .unwrap_or_else(|| "CPUExecutionProvider".to_string());
        let profile = opts.profile;
        Ok(Self {
            card_paths,
            parts: Vec::new(),
            label_map,
            target_spacing: card0.target_spacing,
            data_order: card0.resampling_data_kwargs.order,
            opts,
            active_provider,
            profile,
        })
    }

    pub fn target_spacing(&self) -> [f32; 3] {
        self.target_spacing
    }

    pub fn predict_loaded(
        &mut self,
        data: &Volume<f32>,
        props: &ImageProps,
    ) -> Result<PredictionOutput> {
        self.profile = self.opts.profile;
        let t = StageTimer::new(self.profile);
        let img_t = change_spacing_to_spacing(
            data,
            props.spacing_zyx,
            self.target_spacing,
            self.data_order,
        )
        .map(|v| (*v as i32) as f32);
        t.lap("multi outer_resample_to_model");
        let bbox = bbox_to_nonzero(&img_t);
        t.lap("multi crop_to_nonzero");
        let mut combined = Volume::<u8>::zeros(img_t.shape);
        if self.use_part_workers() {
            let img_crop = img_t.crop(&bbox)?;
            t.lap("multi crop_for_workers");
            self.predict_parts_with_workers(&img_crop, &bbox, &mut combined)?;
            t.lap("multi worker_parts");
        } else if self.keep_parts_resident() {
            self.ensure_parts_loaded()?;
            t.lap("multi ensure_parts_loaded");
            for part in &mut self.parts {
                let seg_local =
                    part.predict_cropped_array(img_t.crop(&bbox)?, self.target_spacing)?;
                t.lap("multi part_predict_array");
                merge_part_labels_at_bbox(
                    &mut combined,
                    &seg_local,
                    &bbox,
                    part.labels(),
                    &self.label_map,
                );
                t.lap("multi merge_local_labels");
            }
        } else {
            for card_path in &self.card_paths {
                let mut part = Segmentator::new(card_path, &self.opts, None)?;
                t.lap("multi build_part");
                let seg_local =
                    part.predict_cropped_array(img_t.crop(&bbox)?, self.target_spacing)?;
                t.lap("multi part_predict_array");
                merge_part_labels_at_bbox(
                    &mut combined,
                    &seg_local,
                    &bbox,
                    part.labels(),
                    &self.label_map,
                );
                t.lap("multi merge_local_labels");
            }
        }
        let seg = change_spacing_to_shape(&combined, data.shape, 0);
        t.lap("multi upsample_to_original");
        Ok((seg, None))
    }

    fn use_part_workers(&self) -> bool {
        self.active_provider == "CUDAExecutionProvider"
            && env::var_os("ONNX_TOTALSEG_CUDA_FULL_RESIDENT").is_none()
    }

    fn keep_parts_resident(&self) -> bool {
        // This CUDA ORT build can segfault when GPU sessions are dropped inside
        // the short-lived CLI. DirectML/CPU can match Python's one-session-at-a-time
        // full path and keep peak memory low.
        self.active_provider == "CUDAExecutionProvider"
    }

    fn ensure_parts_loaded(&mut self) -> Result<()> {
        if self.parts.is_empty() {
            for card_path in &self.card_paths {
                self.parts
                    .push(Segmentator::new(card_path, &self.opts, None)?);
            }
        }
        Ok(())
    }

    fn predict_parts_with_workers(
        &self,
        img_crop: &Volume<f32>,
        bbox: &[[usize; 2]; 3],
        combined: &mut Volume<u8>,
    ) -> Result<()> {
        let t = StageTimer::new(self.profile);
        let temp = tempfile::Builder::new()
            .prefix("onnx_totalseg_full_cuda_")
            .tempdir()?;
        let input_raw = temp.path().join("input.f32");
        write_raw_f32_volume(&input_raw, img_crop)
            .with_context(|| format!("failed to write {}", input_raw.display()))?;
        t.lap("multi worker_write_input");
        let exe = env::current_exe().context("failed to locate current executable")?;
        let shape = shape_arg(img_crop.shape);
        let providers = self.opts.providers.as_ref().map(|p| p.join(","));

        for (idx, card_path) in self.card_paths.iter().enumerate() {
            let output_raw = temp.path().join(format!("part-{idx}.u8"));
            let mut cmd = Command::new(&exe);
            cmd.arg("full-part")
                .arg("--card")
                .arg(card_path)
                .arg("--input-raw")
                .arg(&input_raw)
                .arg("--output-raw")
                .arg(&output_raw)
                .arg("--shape")
                .arg(&shape)
                .arg("--input-cropped")
                .arg("--device")
                .arg(&self.opts.device);
            if let Some(providers) = &providers {
                cmd.arg("--providers").arg(providers);
            }
            if let Some(threads) = self.opts.threads {
                cmd.arg("--threads").arg(threads.to_string());
            }
            if self.opts.fp16 {
                cmd.arg("--fp16");
            } else {
                cmd.arg("--fp32");
            }
            if let Some(ort_lib) = &self.opts.ort_lib {
                cmd.arg("--ort-lib").arg(ort_lib);
            }
            if self.opts.cuda_low_memory {
                cmd.arg("--cuda-low-memory");
            }
            if self.opts.profile {
                cmd.arg("--profile");
            }

            let output = cmd.output().with_context(|| {
                format!(
                    "failed to launch full-part worker for {}",
                    card_path.display()
                )
            })?;
            if self.opts.profile {
                eprint!("{}", String::from_utf8_lossy(&output.stdout));
                eprint!("{}", String::from_utf8_lossy(&output.stderr));
            }
            t.lap("multi worker_run_part");
            if !output.status.success() {
                bail!(
                    "full-part worker failed for {} with status {}\nstdout:\n{}\nstderr:\n{}",
                    card_path.display(),
                    output.status,
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }

            let seg_local = read_raw_u8_volume(&output_raw, img_crop.shape)
                .with_context(|| format!("failed to read {}", output_raw.display()))?;
            t.lap("multi worker_read_part");
            let labels = ModelCard::read(card_path)?.id_to_name();
            merge_part_labels_at_bbox(combined, &seg_local, bbox, labels, &self.label_map);
            t.lap("multi worker_merge_part");
        }

        Ok(())
    }
}

pub enum Predictor {
    Single(Box<Segmentator>),
    Multi(Box<MultiModelSegmentator>),
}

impl Predictor {
    pub fn active_provider(&self) -> &str {
        match self {
            Predictor::Single(s) => &s.active_provider,
            Predictor::Multi(m) => &m.active_provider,
        }
    }

    pub fn target_spacing(&self) -> [f32; 3] {
        match self {
            Predictor::Single(s) => s.target_spacing(),
            Predictor::Multi(m) => m.target_spacing(),
        }
    }

    pub fn preserves_input_orientation(&self) -> bool {
        match self {
            Predictor::Single(s) => s.preserves_input_orientation(),
            Predictor::Multi(_) => false,
        }
    }

    pub fn predict_loaded(
        &mut self,
        data: &Volume<f32>,
        props: &ImageProps,
        lowres: bool,
    ) -> Result<PredictionOutput> {
        match self {
            Predictor::Single(s) => s.predict_loaded(data, props, lowres),
            Predictor::Multi(m) => m.predict_loaded(data, props),
        }
    }

    pub fn predict_loaded_owned(
        &mut self,
        data: Volume<f32>,
        props: &ImageProps,
        lowres: bool,
    ) -> Result<PredictionOutput> {
        match self {
            Predictor::Single(s) => s.predict_loaded_owned(data, props, lowres),
            Predictor::Multi(m) => m.predict_loaded(&data, props),
        }
    }
}

pub fn build_predictor_from_card(
    card_path: &Path,
    opts: RuntimeOptions,
    resample_order: Option<usize>,
) -> Result<Predictor> {
    Ok(Predictor::Single(Box::new(Segmentator::new(
        card_path,
        &opts,
        resample_order,
    )?)))
}

pub fn filter_roi(seg: &mut Volume<u8>, roi: &[u8]) {
    let keep = label_keep_table(roi);
    seg.data.par_iter_mut().for_each(|v| {
        if *v != 0 && !keep[*v as usize] {
            *v = 0;
        }
    });
}

pub fn crop_cascade(
    data: &Volume<f32>,
    props: &ImageProps,
    target: &mut Predictor,
    localizer: &mut Predictor,
    roi_ids: &[u8],
) -> Result<Volume<u8>> {
    crop_with_localizer(
        data,
        props,
        target,
        localizer,
        LocalizerCropOptions {
            crop_ids: roi_ids,
            margin_mm: 20.0,
            output_roi_ids: Some(roi_ids),
            empty_behavior: CropEmptyBehavior::RunTargetFull,
        },
    )
}

#[derive(Clone, Copy, Debug)]
pub enum CropEmptyBehavior {
    RunTargetFull,
    ReturnEmpty,
}

#[derive(Clone, Copy, Debug)]
pub struct LocalizerCropOptions<'a> {
    pub crop_ids: &'a [u8],
    pub margin_mm: f32,
    pub output_roi_ids: Option<&'a [u8]>,
    pub empty_behavior: CropEmptyBehavior,
}

pub fn crop_with_localizer(
    data: &Volume<f32>,
    props: &ImageProps,
    target: &mut Predictor,
    localizer: &mut Predictor,
    options: LocalizerCropOptions<'_>,
) -> Result<Volume<u8>> {
    let (loc, _) = localizer.predict_loaded(data, props, false)?;
    let Some(bbox) = bbox_of_labels(
        &loc,
        options.crop_ids,
        data.shape,
        props.spacing_zyx,
        options.margin_mm,
    ) else {
        if matches!(options.empty_behavior, CropEmptyBehavior::ReturnEmpty) {
            return Ok(Volume::<u8>::zeros(data.shape));
        }
        let (mut seg, _) = target.predict_loaded(data, props, false)?;
        if let Some(ids) = options.output_roi_ids {
            filter_roi(&mut seg, ids);
        }
        return Ok(seg);
    };

    let cropped = data.crop(&bbox)?;
    let (cseg, _) = target.predict_loaded(&cropped, props, false)?;

    if seg_touches_crop_boundary(&cseg, &bbox, data.shape, options.output_roi_ids) {
        eprintln!("[crop] WARNING: mask touches the crop boundary; crop may clip the output mask");
    }

    let mut seg = Volume::<u8>::zeros(data.shape);
    seg.paste(&cseg, &bbox);
    if let Some(ids) = options.output_roi_ids {
        filter_roi(&mut seg, ids);
    }
    Ok(seg)
}

pub fn crop_bbox_from_localizer_mask(
    loc: &Volume<u8>,
    crop_ids: &[u8],
    shape: [usize; 3],
    spacing_zyx: [f32; 3],
    margin_mm: f32,
) -> Option<[[usize; 2]; 3]> {
    bbox_of_labels(loc, crop_ids, shape, spacing_zyx, margin_mm)
}

pub fn read_raw_f32_volume(path: &Path, shape: [usize; 3]) -> Result<Volume<f32>> {
    let expected = shape[0]
        .checked_mul(shape[1])
        .and_then(|n| n.checked_mul(shape[2]))
        .context("raw volume shape overflow")?;
    let expected_bytes = (expected as u64)
        .checked_mul(4)
        .context("raw volume byte length overflow")?;
    let actual = fs::metadata(path)
        .with_context(|| format!("failed to stat {}", path.display()))?
        .len();
    if actual != expected_bytes {
        bail!(
            "raw f32 volume size mismatch for {}: got {} bytes, expected {} bytes for shape {:?}",
            path.display(),
            actual,
            expected_bytes,
            shape
        );
    }

    let file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut data = Vec::with_capacity(expected);
    let mut buf = vec![0_u8; 1024 * 1024];
    let floats_per_chunk = buf.len() / 4;
    let mut remaining = expected;
    while remaining > 0 {
        let floats = remaining.min(floats_per_chunk);
        let bytes = floats * 4;
        reader.read_exact(&mut buf[..bytes])?;
        data.extend(
            buf[..bytes]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        );
        remaining -= floats;
    }
    Volume::from_vec(shape, data)
}

fn write_raw_f32_volume(path: &Path, volume: &Volume<f32>) -> Result<()> {
    let file =
        fs::File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = BufWriter::with_capacity(1024 * 1024, file);
    let mut buf = Vec::with_capacity(1024 * 1024);
    for chunk in volume.data.chunks(256 * 1024) {
        buf.clear();
        for value in chunk {
            buf.extend_from_slice(&value.to_le_bytes());
        }
        writer.write_all(&buf)?;
    }
    writer.flush()?;
    Ok(())
}

fn read_raw_u8_volume(path: &Path, shape: [usize; 3]) -> Result<Volume<u8>> {
    let expected = shape[0]
        .checked_mul(shape[1])
        .and_then(|n| n.checked_mul(shape[2]))
        .context("raw volume shape overflow")?;
    let data = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    if data.len() != expected {
        bail!(
            "raw u8 volume size mismatch for {}: got {} bytes, expected {} bytes for shape {:?}",
            path.display(),
            data.len(),
            expected,
            shape
        );
    }
    Volume::from_vec(shape, data)
}

pub fn write_raw_u8_volume(path: &Path, volume: &Volume<u8>) -> Result<()> {
    let file =
        fs::File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = BufWriter::with_capacity(1024 * 1024, file);
    writer.write_all(&volume.data)?;
    writer.flush()?;
    Ok(())
}

fn shape_arg(shape: [usize; 3]) -> String {
    format!("{},{},{}", shape[0], shape[1], shape[2])
}

fn bbox_of_labels(
    loc: &Volume<u8>,
    roi_ids: &[u8],
    shape: [usize; 3],
    spacing_zyx: [f32; 3],
    margin_mm: f32,
) -> Option<[[usize; 2]; 3]> {
    let keep = label_keep_table(roi_ids);
    let mut lo = [usize::MAX; 3];
    let mut hi = [0usize; 3];
    for z in 0..loc.shape[0] {
        for y in 0..loc.shape[1] {
            for x in 0..loc.shape[2] {
                if keep[*loc.get(z, y, x) as usize] {
                    lo[0] = lo[0].min(z);
                    lo[1] = lo[1].min(y);
                    lo[2] = lo[2].min(x);
                    hi[0] = hi[0].max(z + 1);
                    hi[1] = hi[1].max(y + 1);
                    hi[2] = hi[2].max(x + 1);
                }
            }
        }
    }
    if lo[0] == usize::MAX {
        return None;
    }

    let mut bbox = [[0usize; 2]; 3];
    for axis in 0..3 {
        let margin = ((margin_mm as f64) / (spacing_zyx[axis] as f64)).max(0.0) as usize;
        bbox[axis][0] = lo[axis].saturating_sub(margin);
        bbox[axis][1] = hi[axis].saturating_add(margin).min(shape[axis]);
    }
    Some(bbox)
}

fn seg_touches_crop_boundary(
    seg: &Volume<u8>,
    bbox: &[[usize; 2]; 3],
    full_shape: [usize; 3],
    roi_ids: Option<&[u8]>,
) -> bool {
    let keep = roi_ids.map(label_keep_table);
    let kept = |label: u8| match &keep {
        Some(table) => table[label as usize],
        None => label != 0,
    };
    if bbox[0][0] > 0
        && (0..seg.shape[1]).any(|y| (0..seg.shape[2]).any(|x| kept(*seg.get(0, y, x))))
    {
        return true;
    }
    if bbox[0][1] < full_shape[0]
        && (0..seg.shape[1])
            .any(|y| (0..seg.shape[2]).any(|x| kept(*seg.get(seg.shape[0] - 1, y, x))))
    {
        return true;
    }
    if bbox[1][0] > 0
        && (0..seg.shape[0]).any(|z| (0..seg.shape[2]).any(|x| kept(*seg.get(z, 0, x))))
    {
        return true;
    }
    if bbox[1][1] < full_shape[1]
        && (0..seg.shape[0])
            .any(|z| (0..seg.shape[2]).any(|x| kept(*seg.get(z, seg.shape[1] - 1, x))))
    {
        return true;
    }
    if bbox[2][0] > 0
        && (0..seg.shape[0]).any(|z| (0..seg.shape[1]).any(|y| kept(*seg.get(z, y, 0))))
    {
        return true;
    }
    bbox[2][1] < full_shape[2]
        && (0..seg.shape[0])
            .any(|z| (0..seg.shape[1]).any(|y| kept(*seg.get(z, y, seg.shape[2] - 1))))
}

fn label_keep_table(roi_ids: &[u8]) -> [bool; 256] {
    let mut keep = [false; 256];
    for id in roi_ids {
        keep[*id as usize] = true;
    }
    keep
}

fn merge_part_labels_at_bbox(
    combined: &mut Volume<u8>,
    local: &Volume<u8>,
    bbox: &[[usize; 2]; 3],
    labels: HashMap<u8, String>,
    label_map: &HashMap<String, u8>,
) {
    let mut local_to_global = [0_u8; 256];
    for (local_id, name) in labels {
        if local_id == 0 {
            continue;
        }
        let Some(&global_id) = label_map.get(&name) else {
            continue;
        };
        local_to_global[local_id as usize] = global_id;
    }
    merge_local_labels_at_bbox(combined, local, bbox, &local_to_global);
}

fn merge_local_labels_at_bbox(
    combined: &mut Volume<u8>,
    local: &Volume<u8>,
    bbox: &[[usize; 2]; 3],
    local_to_global: &[u8; 256],
) {
    debug_assert_eq!(
        local.shape,
        [
            bbox[0][1].saturating_sub(bbox[0][0]),
            bbox[1][1].saturating_sub(bbox[1][0]),
            bbox[2][1].saturating_sub(bbox[2][0]),
        ]
    );
    let dst_plane = combined.shape[1] * combined.shape[2];
    let src_plane = local.shape[1] * local.shape[2];
    let z0 = bbox[0][0];
    let z1 = bbox[0][1].min(combined.shape[0]);
    let dst_start = z0 * dst_plane;
    let dst_end = z1 * dst_plane;
    combined.data[dst_start..dst_end]
        .par_chunks_mut(dst_plane)
        .enumerate()
        .for_each(|(local_z, dst_plane_data)| {
            let src_plane_data = &local.data[local_z * src_plane..(local_z + 1) * src_plane];
            for y in 0..local.shape[1] {
                let dst_y = bbox[1][0] + y;
                if dst_y >= combined.shape[1] {
                    continue;
                }
                let src_row = &src_plane_data[y * local.shape[2]..(y + 1) * local.shape[2]];
                let dst_x0 = bbox[2][0];
                if dst_x0 >= combined.shape[2] {
                    continue;
                }
                let len = local.shape[2].min(combined.shape[2] - dst_x0);
                let dst_row = &mut dst_plane_data[dst_y * combined.shape[2] + dst_x0..][..len];
                for (dst, src) in dst_row.iter_mut().zip(src_row.iter().take(len)) {
                    let global_id = local_to_global[*src as usize];
                    if global_id != 0 {
                        *dst = global_id;
                    }
                }
            }
        });
}

fn ct_normalize(data: &mut Volume<f32>, card: &ModelCard) {
    let n = &card.normalization;
    let std = n.std.max(1e-8);
    let lower = n.lower_bound;
    let upper = n.upper_bound;
    let mean = n.mean;
    data.data.par_iter_mut().for_each(|v| {
        *v = v.clamp(lower, upper);
        *v = (*v - mean) / std;
    });
}

fn crop_to_nonzero(data: &Volume<f32>) -> Result<(Volume<f32>, [[usize; 2]; 3])> {
    let bbox = bbox_to_nonzero(data);
    Ok((data.crop(&bbox)?, bbox))
}

fn bbox_to_nonzero(data: &Volume<f32>) -> [[usize; 2]; 3] {
    let plane = data.shape[1] * data.shape[2];
    let nx = data.shape[2];
    let init = ([usize::MAX; 3], [0usize; 3]);
    let (lo, hi) = data
        .data
        .par_chunks(16 * 1024)
        .enumerate()
        .fold(
            || init,
            |(mut lo, mut hi), (chunk_idx, chunk)| {
                let base = chunk_idx * 16 * 1024;
                for (offset, value) in chunk.iter().enumerate() {
                    if *value == 0.0 {
                        continue;
                    }
                    let idx = base + offset;
                    let z = idx / plane;
                    let rem = idx % plane;
                    let y = rem / nx;
                    let x = rem % nx;
                    lo[0] = lo[0].min(z);
                    lo[1] = lo[1].min(y);
                    lo[2] = lo[2].min(x);
                    hi[0] = hi[0].max(z + 1);
                    hi[1] = hi[1].max(y + 1);
                    hi[2] = hi[2].max(x + 1);
                }
                (lo, hi)
            },
        )
        .reduce(
            || init,
            |(mut lo_a, mut hi_a), (lo_b, hi_b)| {
                for axis in 0..3 {
                    lo_a[axis] = lo_a[axis].min(lo_b[axis]);
                    hi_a[axis] = hi_a[axis].max(hi_b[axis]);
                }
                (lo_a, hi_a)
            },
        );

    if lo[0] == usize::MAX {
        return full_bbox(data.shape);
    }
    [[lo[0], hi[0]], [lo[1], hi[1]], [lo[2], hi[2]]]
}

fn full_bbox(shape: [usize; 3]) -> [[usize; 2]; 3] {
    [[0, shape[0]], [0, shape[1]], [0, shape[2]]]
}

fn pad_to_patch(data: &Volume<f32>, patch: [usize; 3]) -> (Volume<f32>, [[usize; 2]; 3]) {
    let new_shape = [
        data.shape[0].max(patch[0]),
        data.shape[1].max(patch[1]),
        data.shape[2].max(patch[2]),
    ];
    let below = [
        (new_shape[0] - data.shape[0]) / 2,
        (new_shape[1] - data.shape[1]) / 2,
        (new_shape[2] - data.shape[2]) / 2,
    ];
    let revert = [
        [below[0], below[0] + data.shape[0]],
        [below[1], below[1] + data.shape[1]],
        [below[2], below[2] + data.shape[2]],
    ];
    let mut out = Volume::<f32>::zeros(new_shape);
    out.paste(data, &revert);
    (out, revert)
}

fn compute_steps_for_sliding_window(
    image: [usize; 3],
    tile: [usize; 3],
    tile_step_size: f32,
) -> Result<[Vec<usize>; 3]> {
    if !(0.0..=1.0).contains(&tile_step_size) || tile_step_size == 0.0 {
        bail!("tile_step_size must be in (0, 1]");
    }
    if image.iter().zip(tile).any(|(i, t)| *i < t) {
        bail!(
            "image size {:?} is smaller than patch size {:?}",
            image,
            tile
        );
    }
    let mut out: [Vec<usize>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for d in 0..3 {
        let target_step = tile[d] as f32 * tile_step_size;
        let num = (((image[d] - tile[d]) as f32 / target_step).ceil() as usize) + 1;
        let max_step = image[d] - tile[d];
        let actual = if num > 1 {
            max_step as f64 / (num - 1) as f64
        } else {
            99_999_999_999.0
        };
        out[d] = (0..num)
            .map(|i| round_half_to_even_usize(actual * i as f64))
            .collect();
    }
    Ok(out)
}

fn round_half_to_even_usize(v: f64) -> usize {
    let floor = v.floor();
    let frac = v - floor;
    let rounded = if (frac - 0.5).abs() < 1e-9 {
        let floor_i = floor as usize;
        if floor_i.is_multiple_of(2) {
            floor
        } else {
            floor + 1.0
        }
    } else {
        v.round()
    };
    rounded.max(0.0) as usize
}

fn copy_patch_rows(padded: &Volume<f32>, origin: [usize; 3], patch: [usize; 3], out: &mut [f32]) {
    for z in 0..patch[0] {
        for y in 0..patch[1] {
            let src = padded.idx(origin[0] + z, origin[1] + y, origin[2]);
            let dst = (z * patch[1] + y) * patch[2];
            out[dst..dst + patch[2]].copy_from_slice(&padded.data[src..src + patch[2]]);
        }
    }
}

fn inference_mirror_combinations(card: &ModelCard) -> Result<Vec<[bool; 3]>> {
    if !card.inference.use_mirroring.unwrap_or(false) {
        return Ok(vec![[false; 3]]);
    }
    let axes = card
        .inference
        .mirror_axes
        .clone()
        .unwrap_or_else(|| vec![0, 1, 2]);
    let mut seen = [false; 3];
    for &axis in &axes {
        if axis >= 3 {
            bail!("mirror axis {axis} is invalid for a 3D model");
        }
        if seen[axis] {
            bail!("mirror axis {axis} is duplicated");
        }
        seen[axis] = true;
    }
    let mut combinations = Vec::with_capacity(1usize << axes.len());
    for mask in 0..(1usize << axes.len()) {
        let mut flips = [false; 3];
        for (bit, &axis) in axes.iter().enumerate() {
            flips[axis] = mask & (1usize << bit) != 0;
        }
        combinations.push(flips);
    }
    Ok(combinations)
}

fn mirror_patch_input(src: &[f32], dst: &mut [f32], patch: [usize; 3], flips: [bool; 3]) {
    for z in 0..patch[0] {
        let src_z = if flips[0] { patch[0] - 1 - z } else { z };
        for y in 0..patch[1] {
            let src_y = if flips[1] { patch[1] - 1 - y } else { y };
            for x in 0..patch[2] {
                let src_x = if flips[2] { patch[2] - 1 - x } else { x };
                dst[(z * patch[1] + y) * patch[2] + x] =
                    src[(src_z * patch[1] + src_y) * patch[2] + src_x];
            }
        }
    }
}

fn accumulate_unmirrored_scores(
    dst: &mut [f32],
    src: &[f32],
    heads: usize,
    patch: [usize; 3],
    flips: [bool; 3],
) {
    let patch_n = patch[0] * patch[1] * patch[2];
    for head in 0..heads {
        for z in 0..patch[0] {
            let src_z = if flips[0] { patch[0] - 1 - z } else { z };
            for y in 0..patch[1] {
                let src_y = if flips[1] { patch[1] - 1 - y } else { y };
                for x in 0..patch[2] {
                    let src_x = if flips[2] { patch[2] - 1 - x } else { x };
                    let dst_i = head * patch_n + (z * patch[1] + y) * patch[2] + x;
                    let src_i = head * patch_n + (src_z * patch[1] + src_y) * patch[2] + src_x;
                    dst[dst_i] += src[src_i];
                }
            }
        }
    }
}

fn accumulate_unmirrored_scores_view(
    dst: &mut [f32],
    src: &ArrayViewD<'_, f32>,
    heads: usize,
    patch: [usize; 3],
    flips: [bool; 3],
) {
    let patch_n = patch[0] * patch[1] * patch[2];
    for head in 0..heads {
        for z in 0..patch[0] {
            let src_z = if flips[0] { patch[0] - 1 - z } else { z };
            for y in 0..patch[1] {
                let src_y = if flips[1] { patch[1] - 1 - y } else { y };
                for x in 0..patch[2] {
                    let src_x = if flips[2] { patch[2] - 1 - x } else { x };
                    let dst_i = head * patch_n + (z * patch[1] + y) * patch[2] + x;
                    dst[dst_i] += src[[0, head, src_z, src_y, src_x]];
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_patch_logits(
    logits: &mut [f32],
    pred: &[f32],
    gaussian: Option<&[f32]>,
    heads: usize,
    spatial: usize,
    padded_shape: [usize; 3],
    patch: [usize; 3],
    origin: [usize; 3],
) {
    let patch_n = patch[0] * patch[1] * patch[2];
    logits
        .par_chunks_mut(spatial)
        .zip(pred.par_chunks(patch_n))
        .take(heads)
        .for_each(|(logits_head, pred_head)| {
            for z in 0..patch[0] {
                for y in 0..patch[1] {
                    let local = (z * patch[1] + y) * patch[2];
                    let dst = ((origin[0] + z) * padded_shape[1] + origin[1] + y) * padded_shape[2]
                        + origin[2];
                    let pred_row = &pred_head[local..local + patch[2]];
                    let logits_row = &mut logits_head[dst..dst + patch[2]];
                    if let Some(g) = gaussian {
                        let gauss_row = &g[local..local + patch[2]];
                        for ((dst, src), w) in
                            logits_row.iter_mut().zip(pred_row.iter()).zip(gauss_row)
                        {
                            *dst += *src * *w;
                        }
                    } else {
                        for (dst, src) in logits_row.iter_mut().zip(pred_row.iter()) {
                            *dst += *src;
                        }
                    }
                }
            }
        });
}

#[allow(clippy::too_many_arguments)]
fn accumulate_patch_logits_view(
    logits: &mut [f32],
    pred: &ArrayViewD<'_, f32>,
    gaussian: Option<&[f32]>,
    heads: usize,
    spatial: usize,
    padded_shape: [usize; 3],
    patch: [usize; 3],
    origin: [usize; 3],
) {
    for h in 0..heads {
        let logits_head = &mut logits[h * spatial..(h + 1) * spatial];
        for z in 0..patch[0] {
            for y in 0..patch[1] {
                let local = (z * patch[1] + y) * patch[2];
                let dst = ((origin[0] + z) * padded_shape[1] + origin[1] + y) * padded_shape[2]
                    + origin[2];
                let logits_row = &mut logits_head[dst..dst + patch[2]];
                if let Some(g) = gaussian {
                    let gauss_row = &g[local..local + patch[2]];
                    for x in 0..patch[2] {
                        logits_row[x] += pred[[0, h, z, y, x]] * gauss_row[x];
                    }
                } else {
                    for x in 0..patch[2] {
                        logits_row[x] += pred[[0, h, z, y, x]];
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayView, IxDyn};

    #[test]
    fn copy_patch_rows_copies_contiguous_patch() {
        let data = (0..24).map(|v| v as f32).collect::<Vec<_>>();
        let volume = Volume::from_vec([2, 3, 4], data).expect("valid test volume");
        let mut patch = vec![0.0; 12];
        copy_patch_rows(&volume, [0, 1, 1], [2, 2, 3], &mut patch);
        assert_eq!(
            patch,
            vec![
                5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 17.0, 18.0, 19.0, 21.0, 22.0, 23.0
            ]
        );
    }

    #[test]
    fn mirrored_patch_scores_restore_original_coordinates() {
        let patch = [2, 2, 3];
        let src = (0..12).map(|value| value as f32).collect::<Vec<_>>();
        let flips = [true, false, true];
        let mut mirrored = vec![0.0_f32; src.len()];
        mirror_patch_input(&src, &mut mirrored, patch, flips);
        let mut restored = vec![0.0_f32; src.len()];
        accumulate_unmirrored_scores(&mut restored, &mirrored, 1, patch, flips);
        assert_eq!(restored, src);
    }

    #[test]
    fn contiguous_and_view_accumulators_match() {
        let heads = 2;
        let spatial = 12;
        let padded_shape = [1, 3, 4];
        let patch = [1, 2, 3];
        let origin = [0, 1, 1];
        let pred = (1..=heads * patch[0] * patch[1] * patch[2])
            .map(|v| v as f32)
            .collect::<Vec<_>>();
        let gaussian = vec![1.0, 0.5, 2.0, 3.0, 4.0, 5.0];

        let mut logits_contiguous = vec![0.0; heads * spatial];
        accumulate_patch_logits(
            &mut logits_contiguous,
            &pred,
            Some(&gaussian),
            heads,
            spatial,
            padded_shape,
            patch,
            origin,
        );

        let pred_view =
            ArrayView::from_shape(IxDyn(&[1, heads, patch[0], patch[1], patch[2]]), &pred)
                .expect("valid test logits view");
        let mut logits_view = vec![0.0; heads * spatial];
        accumulate_patch_logits_view(
            &mut logits_view,
            &pred_view,
            Some(&gaussian),
            heads,
            spatial,
            padded_shape,
            patch,
            origin,
        );

        assert_eq!(logits_contiguous, logits_view);
    }

    #[test]
    fn sliding_window_steps_use_numpy_rounding() {
        let steps = compute_steps_for_sliding_window([163, 117, 117], [64, 64, 64], 0.5)
            .expect("valid sliding-window shape");
        assert_eq!(steps[0], vec![0, 25, 50, 74, 99]);
        assert_eq!(steps[1], vec![0, 26, 53]);
        assert_eq!(steps[2], vec![0, 26, 53]);
    }

    #[test]
    fn merge_local_labels_preserves_previous_parts_and_overrides_hits() {
        let mut combined = Volume::from_vec(
            [1, 3, 6],
            vec![
                0_u8, 0, 0, 0, 0, 0, //
                0, 10, 0, 30, 0, 0, //
                0, 4, 0, 6, 0, 0,
            ],
        )
        .unwrap();
        let local = Volume::from_vec([1, 2, 4], vec![1_u8, 0, 2, 3, 0, 4, 2, 0]).unwrap();
        let mut map = [0_u8; 256];
        map[1] = 101;
        map[2] = 102;
        map[4] = 104;

        merge_local_labels_at_bbox(&mut combined, &local, &[[0, 1], [1, 3], [1, 5]], &map);

        assert_eq!(
            combined.data,
            vec![
                0, 0, 0, 0, 0, 0, //
                0, 101, 0, 102, 0, 0, //
                0, 4, 104, 102, 0, 0,
            ]
        );
    }

    #[test]
    fn mixed_provider_order_requires_cpu_last() {
        let err = validate_provider_order(&[
            "CPUExecutionProvider".to_string(),
            "CUDAExecutionProvider".to_string(),
        ])
        .expect_err("CPU-first mixed provider list must fail");

        assert!(
            err.to_string()
                .contains("CPUExecutionProvider can only appear"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn mixed_provider_order_accepts_cpu_fallback() {
        validate_provider_order(&[
            "CUDAExecutionProvider".to_string(),
            "CPUExecutionProvider".to_string(),
        ])
        .expect("CPU fallback at the end is valid");
        validate_provider_order(&["CPUExecutionProvider".to_string()])
            .expect("CPU-only provider list is valid");
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct GaussianKey {
    patch: [usize; 3],
    sigma_scale_bits: u32,
}

fn cached_gaussian(patch: [usize; 3], sigma_scale: f32) -> Arc<[f32]> {
    static CACHE: OnceLock<Mutex<HashMap<GaussianKey, Arc<[f32]>>>> = OnceLock::new();
    let key = GaussianKey {
        patch,
        sigma_scale_bits: sigma_scale.to_bits(),
    };
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().expect("gaussian cache mutex poisoned");
    if let Some(gaussian) = guard.get(&key) {
        return Arc::clone(gaussian);
    }
    let gaussian = Arc::<[f32]>::from(compute_gaussian(patch, sigma_scale));
    guard.insert(key, Arc::clone(&gaussian));
    gaussian
}

fn compute_gaussian(patch: [usize; 3], sigma_scale: f32) -> Vec<f32> {
    fn axis_weights(len: usize, sigma_scale: f32) -> Vec<f32> {
        let center = (len / 2) as f32;
        let sigma = len as f32 * sigma_scale;
        let denom = 2.0 * sigma * sigma;
        (0..len)
            .map(|i| {
                let d = i as f32 - center;
                (-(d * d / denom)).exp()
            })
            .collect()
    }

    let z_weights = axis_weights(patch[0], sigma_scale);
    let y_weights = axis_weights(patch[1], sigma_scale);
    let x_weights = axis_weights(patch[2], sigma_scale);
    let mut out = vec![0.0_f32; patch[0] * patch[1] * patch[2]];
    let mut max_v = 0.0_f32;
    for (z, &z_weight) in z_weights.iter().enumerate().take(patch[0]) {
        for (y, &y_weight) in y_weights.iter().enumerate().take(patch[1]) {
            for (x, &x_weight) in x_weights.iter().enumerate().take(patch[2]) {
                let v = z_weight * y_weight * x_weight;
                let i = (z * patch[1] + y) * patch[2] + x;
                out[i] = v;
                max_v = max_v.max(v);
            }
        }
    }
    let mut min_nonzero = f32::INFINITY;
    for v in &mut out {
        *v /= max_v;
        if *v > 0.0 {
            min_nonzero = min_nonzero.min(*v);
        }
    }
    for v in &mut out {
        if *v == 0.0 {
            *v = min_nonzero;
        }
    }
    out
}

fn accumulate_patch_weights(
    weights: &mut [f32],
    gaussian: Option<&[f32]>,
    padded_shape: [usize; 3],
    patch: [usize; 3],
    origin: [usize; 3],
) {
    for z in 0..patch[0] {
        for y in 0..patch[1] {
            let local = (z * patch[1] + y) * patch[2];
            let dst =
                ((origin[0] + z) * padded_shape[1] + origin[1] + y) * padded_shape[2] + origin[2];
            let weights_row = &mut weights[dst..dst + patch[2]];
            if let Some(g) = gaussian {
                let gauss_row = &g[local..local + patch[2]];
                for (dst, w) in weights_row.iter_mut().zip(gauss_row) {
                    *dst += *w;
                }
            } else {
                for dst in weights_row {
                    *dst += 1.0;
                }
            }
        }
    }
}

fn normalize_logits_by_weights(logits: &mut [f32], weights: &[f32], heads: usize, spatial: usize) {
    logits
        .par_chunks_mut(spatial)
        .take(heads)
        .for_each(|logits_head| {
            for (v, w) in logits_head.iter_mut().zip(weights) {
                if *w != 0.0 {
                    *v /= *w;
                }
            }
        });
}

fn argmax_single_patch_logits(
    pred: &[f32],
    heads: usize,
    patch: [usize; 3],
    shape: [usize; 3],
    revert: [[usize; 2]; 3],
) -> Volume<u8> {
    let patch_n = patch[0] * patch[1] * patch[2];
    let out_plane = shape[1] * shape[2];
    let mut out = vec![0_u8; shape[0] * out_plane];
    out.par_chunks_mut(out_plane)
        .enumerate()
        .for_each(|(z, out_plane_data)| {
            let pz = revert[0][0] + z;
            for y in 0..shape[1] {
                let py = revert[1][0] + y;
                for x in 0..shape[2] {
                    let px = revert[2][0] + x;
                    let pidx = (pz * patch[1] + py) * patch[2] + px;
                    let mut best_h = 0usize;
                    let mut best_v = pred[pidx];
                    for h in 1..heads {
                        let v = pred[h * patch_n + pidx];
                        if v > best_v {
                            best_v = v;
                            best_h = h;
                        }
                    }
                    out_plane_data[y * shape[2] + x] = best_h as u8;
                }
            }
        });
    Volume::from_vec(shape, out).expect("single-patch argmax volume shape is valid")
}

fn argmax_single_patch_logits_view(
    pred: &ArrayViewD<'_, f32>,
    heads: usize,
    _patch: [usize; 3],
    shape: [usize; 3],
    revert: [[usize; 2]; 3],
) -> Volume<u8> {
    let out_plane = shape[1] * shape[2];
    let mut out = vec![0_u8; shape[0] * out_plane];
    out.par_chunks_mut(out_plane)
        .enumerate()
        .for_each(|(z, out_plane_data)| {
            let pz = revert[0][0] + z;
            for y in 0..shape[1] {
                let py = revert[1][0] + y;
                for x in 0..shape[2] {
                    let px = revert[2][0] + x;
                    let mut best_h = 0usize;
                    let mut best_v = pred[[0, 0, pz, py, px]];
                    for h in 1..heads {
                        let v = pred[[0, h, pz, py, px]];
                        if v > best_v {
                            best_v = v;
                            best_h = h;
                        }
                    }
                    out_plane_data[y * shape[2] + x] = best_h as u8;
                }
            }
        });
    Volume::from_vec(shape, out).expect("single-patch argmax volume shape is valid")
}

fn argmax_logits(logits: &[f32], heads: usize, shape: [usize; 3]) -> Volume<u8> {
    let spatial = shape[0] * shape[1] * shape[2];
    let mut out = vec![0_u8; spatial];
    out.par_iter_mut().enumerate().for_each(|(i, dst)| {
        let mut best_h = 0usize;
        let mut best_v = logits[i];
        for h in 1..heads {
            let v = logits[h * spatial + i];
            if v > best_v {
                best_v = v;
                best_h = h;
            }
        }
        *dst = best_h as u8;
    });
    Volume::from_vec(shape, out).expect("argmax volume shape is valid")
}

fn resize_logits_to_shape(
    logits: &[f32],
    heads: usize,
    from_shape: [usize; 3],
    to_shape: [usize; 3],
    from_spacing: [f32; 3],
    to_spacing: [f32; 3],
) -> Result<Vec<f32>> {
    let spatial_from = from_shape[0] * from_shape[1] * from_shape[2];
    let spatial_to = to_shape[0] * to_shape[1] * to_shape[2];
    if logits.len() != heads * spatial_from {
        bail!(
            "logit buffer size mismatch: got {}, expected {} for heads={} shape={:?}",
            logits.len(),
            heads * spatial_from,
            heads,
            from_shape
        );
    }
    let per_head: Result<Vec<Vec<f32>>> = (0..heads)
        .into_par_iter()
        .map(|head| {
            let start = head * spatial_from;
            let vol = Volume::from_vec(from_shape, logits[start..start + spatial_from].to_vec())?;
            let resized =
                resample_data_or_seg_to_shape(&vol, to_shape, from_spacing, to_spacing, 1, 0, None);
            Ok(resized.data)
        })
        .collect();
    let mut out = vec![0.0_f32; heads * spatial_to];
    for (head, data) in per_head?.into_iter().enumerate() {
        out[head * spatial_to..(head + 1) * spatial_to].copy_from_slice(&data);
    }
    Ok(out)
}

fn unpad_logits(
    logits: &[f32],
    heads: usize,
    padded_shape: [usize; 3],
    original_shape: [usize; 3],
    revert: [[usize; 2]; 3],
) -> Vec<f32> {
    let spatial_p = padded_shape[0] * padded_shape[1] * padded_shape[2];
    let spatial_o = original_shape[0] * original_shape[1] * original_shape[2];
    let mut out = vec![0.0_f32; heads * spatial_o];
    out.par_chunks_mut(spatial_o)
        .enumerate()
        .for_each(|(h, out_head)| {
            for z in 0..original_shape[0] {
                for y in 0..original_shape[1] {
                    for x in 0..original_shape[2] {
                        let src = ((revert[0][0] + z) * padded_shape[1] + (revert[1][0] + y))
                            * padded_shape[2]
                            + (revert[2][0] + x);
                        let dst = (z * original_shape[1] + y) * original_shape[2] + x;
                        out_head[dst] = logits[h * spatial_p + src];
                    }
                }
            }
        });
    out
}

fn ensure_ort(opts: &RuntimeOptions) -> Result<()> {
    static INIT: OnceLock<Result<(), String>> = OnceLock::new();
    INIT.get_or_init(|| {
        let lib = opts
            .ort_lib
            .clone()
            .or_else(|| env::var_os("ONNX_TOTALSEG_ORT_LIB").map(PathBuf::from));
        match lib {
            Some(path) => ort::init_from(&path)
                .map_err(|e| format!("failed to load ONNX Runtime from {}: {e}", path.display()))?
                .with_name("onnx-totalseg")
                .with_telemetry(false)
                .commit()
                .then_some(())
                .ok_or_else(|| "ONNX Runtime environment was already initialized".to_string()),
            None => ort::init()
                .with_name("onnx-totalseg")
                .with_telemetry(false)
                .commit()
                .then_some(())
                .ok_or_else(|| "ONNX Runtime environment was already initialized".to_string()),
        }
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

pub fn resolve_provider_names(
    providers: Option<&[String]>,
    device: &str,
    ort_lib: Option<PathBuf>,
) -> Result<Vec<String>> {
    let probe = RuntimeOptions {
        providers: None,
        device: "cpu".to_string(),
        threads: None,
        fp16: false,
        int8: false,
        ort_lib: ort_lib.clone(),
        profile: false,
        low_memory_cpu: false,
        cuda_low_memory: false,
        lazy_session: false,
    };
    ensure_ort(&probe)?;
    if let Some(providers) = providers {
        let mut names = Vec::new();
        for provider in providers {
            if is_windows_ml_group_alias(provider) {
                names.extend(resolve_windows_ml_provider_names(ort_lib.as_deref())?);
            } else {
                names.push(canonical_provider_name(provider)?);
            }
        }
        validate_provider_order(&names)?;
        if names
            .iter()
            .any(|name| windows_ml::is_catalog_provider_name(name))
        {
            windows_ml::register_byo_provider_libraries()?;
            windows_ml::register_installed_certified(ort_lib.as_deref())?;
        }
        for name in &names {
            if !provider_is_available(name)? {
                bail!(
                    "requested execution provider {name} is not available in this ONNX Runtime build; available supported providers: {}",
                    available_provider_summary()?
                );
            }
        }
        return Ok(names);
    }

    let device = device.to_ascii_lowercase();
    let device = if device == "gpu" {
        "directml".to_string()
    } else {
        device
    };

    if device == "cpu" {
        return Ok(vec!["CPUExecutionProvider".to_string()]);
    }
    if device == "auto" {
        let mut out = Vec::new();
        if provider_is_available("DmlExecutionProvider")? {
            out.push("DmlExecutionProvider".to_string());
        }
        out.push("CPUExecutionProvider".to_string());
        return Ok(out);
    }
    if is_windows_ml_group_alias(&device) {
        let mut out = resolve_windows_ml_provider_names(ort_lib.as_deref())?;
        out.push("CPUExecutionProvider".to_string());
        return Ok(out);
    }

    let name = device_to_provider_name(&device)?;
    if windows_ml::is_catalog_provider_name(name) {
        windows_ml::register_byo_provider_libraries()?;
        windows_ml::register_installed_certified(ort_lib.as_deref())?;
    }
    if !provider_is_available(name)? {
        bail!(
            "requested device '{device}' -> {name} is not available in this ONNX Runtime build; available supported providers: {}",
            available_provider_summary()?
        );
    }
    Ok(vec![name.to_string(), "CPUExecutionProvider".to_string()])
}

fn validate_provider_order(names: &[String]) -> Result<()> {
    if names.len() <= 1 || names.iter().all(|name| name == "CPUExecutionProvider") {
        return Ok(());
    }
    if names[..names.len() - 1]
        .iter()
        .any(|name| name == "CPUExecutionProvider")
    {
        bail!(
            "CPUExecutionProvider can only appear as the final fallback in a mixed provider list; use --device cpu for CPU-only or put CPU last, for example --providers cuda,cpu"
        );
    }
    Ok(())
}

pub fn is_gpu_provider(provider: &str) -> bool {
    // OpenVINO here targets the CPU device, so it is CPU-class for the purposes of
    // fp16 selection and short-lived GPU worker processes (both keyed off "is GPU").
    !matches!(
        provider,
        "CPUExecutionProvider" | "OpenVINOExecutionProvider"
    )
}

pub fn provider_plan_uses_gpu(providers: &[String]) -> bool {
    providers
        .first()
        .is_some_and(|provider| is_gpu_provider(provider))
}

fn execution_providers_from_names(
    names: &[String],
    low_memory_cpu: bool,
    cuda_low_memory: bool,
) -> Result<Vec<ep::ExecutionProviderDispatch>> {
    let mut eps = Vec::new();
    let cpu_only = names.iter().all(|name| {
        matches!(
            name.to_ascii_lowercase().as_str(),
            "cpu" | "cpuexecutionprovider"
        )
    });
    for name in names {
        let lower = name.to_ascii_lowercase();
        let ep = match lower.as_str() {
            "cpu" | "cpuexecutionprovider" => {
                if low_memory_cpu && cpu_only {
                    ep::CPU::default()
                        .with_arena_allocator(false)
                        .build()
                        .error_on_failure()
                } else {
                    continue;
                }
            }
            "cuda" | "cudaexecutionprovider" => {
                let mut cuda = ep::CUDA::default();
                let low_memory =
                    cuda_low_memory || env::var_os("ONNX_TOTALSEG_CUDA_LOW_MEMORY").is_some();
                if low_memory {
                    cuda = cuda
                        .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
                        .with_conv_algorithm_search(ConvAlgorithmSearch::Heuristic)
                        .with_conv_max_workspace(false);
                }
                if let Ok(value) = env::var("ONNX_TOTALSEG_CUDA_CONV_SEARCH") {
                    cuda = cuda.with_conv_algorithm_search(
                        match value.to_ascii_lowercase().as_str() {
                            "exhaustive" => ConvAlgorithmSearch::Exhaustive,
                            "heuristic" => ConvAlgorithmSearch::Heuristic,
                            "default" => ConvAlgorithmSearch::Default,
                            other => bail!("unsupported ONNX_TOTALSEG_CUDA_CONV_SEARCH='{other}'"),
                        },
                    );
                }
                if let Ok(value) = env::var("ONNX_TOTALSEG_CUDA_CONV_MAX_WORKSPACE") {
                    let enable = match value.to_ascii_lowercase().as_str() {
                        "1" | "true" | "yes" | "on" => true,
                        "0" | "false" | "no" | "off" => false,
                        other => {
                            bail!("unsupported ONNX_TOTALSEG_CUDA_CONV_MAX_WORKSPACE='{other}'")
                        }
                    };
                    cuda = cuda.with_conv_max_workspace(enable);
                }
                if let Ok(value) = env::var("ONNX_TOTALSEG_CUDA_ARENA_STRATEGY") {
                    cuda = cuda.with_arena_extend_strategy(
                        match value.to_ascii_lowercase().as_str() {
                            "same" | "same_as_requested" | "sameasrequested" => {
                                ArenaExtendStrategy::SameAsRequested
                            }
                            "power2" | "next_power_of_two" | "nextpoweroftwo" => {
                                ArenaExtendStrategy::NextPowerOfTwo
                            }
                            other => {
                                bail!("unsupported ONNX_TOTALSEG_CUDA_ARENA_STRATEGY='{other}'")
                            }
                        },
                    );
                }
                if let Ok(value) = env::var("ONNX_TOTALSEG_CUDA_MEM_LIMIT_MB") {
                    let mb = value.parse::<usize>().with_context(|| {
                        format!("invalid ONNX_TOTALSEG_CUDA_MEM_LIMIT_MB='{value}'")
                    })?;
                    cuda = cuda.with_memory_limit(mb * 1024 * 1024);
                }
                cuda.build().error_on_failure()
            }
            "directml" | "directmlexecutionprovider" => {
                ep::DirectML::default().build().error_on_failure()
            }
            "openvino" | "openvinoexecutionprovider" => {
                // Self-bundled OpenVINO runtime, CPU device. oneDNN-backed int8/fp32
                // 3D-conv kernels; ~2-2.7x over MLAS on the CPU tiers.
                let mut ov = ep::OpenVINO::default().with_device_type("CPU");
                if env::var_os("ONNX_TOTALSEG_OPENVINO_QDQ_OPTIMIZER").is_some() {
                    ov = ov.with_qdq_optimizer(true);
                }
                if let Ok(cache_dir) = env::var("ONNX_TOTALSEG_OPENVINO_CACHE_DIR") {
                    ov = ov.with_cache_dir(cache_dir);
                }
                let threads = env::var("ONNX_TOTALSEG_OPENVINO_THREADS")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok());
                let precision = env::var("ONNX_TOTALSEG_OPENVINO_INFERENCE_PRECISION")
                    .ok()
                    .map(|value| {
                        Ok(match value.to_ascii_lowercase().as_str() {
                            "f32" | "fp32" => "f32",
                            "bf16" => "bf16",
                            other => bail!(
                                "unsupported ONNX_TOTALSEG_OPENVINO_INFERENCE_PRECISION='{other}'"
                            ),
                        })
                    })
                    .transpose()?;
                let execution_mode = env::var("ONNX_TOTALSEG_OPENVINO_EXECUTION_MODE")
                    .ok()
                    .map(|value| {
                        Ok(match value.to_ascii_lowercase().as_str() {
                            "performance" => "PERFORMANCE",
                            "accuracy" => "ACCURACY",
                            other => {
                                bail!("unsupported ONNX_TOTALSEG_OPENVINO_EXECUTION_MODE='{other}'")
                            }
                        })
                    })
                    .transpose()?;
                if precision.is_some() || execution_mode.is_some() {
                    let mut cpu_config = serde_json::Map::new();
                    if let Some(precision) = precision {
                        cpu_config.insert(
                            "INFERENCE_PRECISION_HINT".to_string(),
                            serde_json::Value::String(precision.to_string()),
                        );
                    }
                    if let Some(execution_mode) = execution_mode {
                        cpu_config.insert(
                            "EXECUTION_MODE_HINT".to_string(),
                            serde_json::Value::String(execution_mode.to_string()),
                        );
                    }
                    if let Some(threads) = threads {
                        cpu_config.insert(
                            "INFERENCE_NUM_THREADS".to_string(),
                            serde_json::Value::String(threads.to_string()),
                        );
                    }
                    let config = serde_json::json!({ "CPU": cpu_config }).to_string();
                    ov = ov.with_arbitrary_config("load_config", config);
                } else if let Some(threads) = threads {
                    ov = ov.with_num_threads(threads);
                }
                ov.build().error_on_failure()
            }
            other if windows_ml::is_catalog_provider_name(other) => continue,
            other => bail!("unsupported execution provider '{other}'"),
        };
        eps.push(ep);
    }
    Ok(eps)
}

fn attach_windows_ml_devices(
    mut builder: ort::session::builder::SessionBuilder,
    names: &[String],
) -> Result<ort::session::builder::SessionBuilder> {
    let catalog_names = names
        .iter()
        .filter(|name| windows_ml::is_catalog_provider_name(name))
        .collect::<Vec<_>>();
    if catalog_names.is_empty() {
        return Ok(builder);
    }

    let env = Environment::current().map_err(|e| anyhow::anyhow!("{e}"))?;
    for name in catalog_names {
        let mut matched = 0usize;
        builder = builder
            .with_devices(
                env.devices().filter(|device| {
                    let is_match = device
                        .ep()
                        .is_ok_and(|ep| windows_ml::provider_name_eq(ep, name));
                    if is_match {
                        matched += 1;
                    }
                    is_match
                }),
                None,
            )
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        if matched == 0 {
            bail!("Windows ML execution provider {name} has no registered devices");
        }
    }
    Ok(builder)
}

fn canonical_provider_name(name: &str) -> Result<String> {
    let lower = name.trim().to_ascii_lowercase();
    Ok(match lower.as_str() {
        "cpu" | "cpuexecutionprovider" => "CPUExecutionProvider",
        "cuda" | "cudaexecutionprovider" => "CUDAExecutionProvider",
        "directml" | "dml" | "dmlexecutionprovider" | "directmlexecutionprovider" => {
            "DmlExecutionProvider"
        }
        "qnn" | "windowsml-qnn" | "winml-qnn" | "qnnexecutionprovider" => "QNNExecutionProvider",
        "openvino" | "windowsml-openvino" | "winml-openvino" | "openvinoexecutionprovider" => {
            "OpenVINOExecutionProvider"
        }
        "vitis"
        | "vitisai"
        | "windowsml-vitis"
        | "windowsml-vitisai"
        | "winml-vitis"
        | "winml-vitisai"
        | "vitisexecutionprovider"
        | "vitisaiexecutionprovider" => "VitisAIExecutionProvider",
        "migraphx" | "windowsml-migraphx" | "winml-migraphx" | "migraphxexecutionprovider" => {
            "MIGraphXExecutionProvider"
        }
        "nvrtx"
        | "nvtensorrtrtx"
        | "nvtensorrtx"
        | "windowsml-nvrtx"
        | "windowsml-nvidia"
        | "winml-nvrtx"
        | "winml-nvidia"
        | "nvtensorrtrtxexecutionprovider"
        | "nvtensorrtxexecutionprovider" => "NvTensorRtRtxExecutionProvider",
        other => bail!("unsupported execution provider '{other}'"),
    }
    .to_string())
}

fn device_to_provider_name(device: &str) -> Result<&'static str> {
    Ok(match device {
        "cuda" => "CUDAExecutionProvider",
        "directml" | "dml" => "DmlExecutionProvider",
        "qnn" | "windowsml-qnn" | "winml-qnn" => "QNNExecutionProvider",
        "openvino" | "windowsml-openvino" | "winml-openvino" => "OpenVINOExecutionProvider",
        "vitis" | "vitisai" | "windowsml-vitis" | "windowsml-vitisai" | "winml-vitis"
        | "winml-vitisai" => "VitisAIExecutionProvider",
        "migraphx" | "windowsml-migraphx" | "winml-migraphx" => "MIGraphXExecutionProvider",
        "nvrtx" | "nvtensorrtrtx" | "nvtensorrtx" | "windowsml-nvrtx" | "windowsml-nvidia"
        | "winml-nvrtx" | "winml-nvidia" => "NvTensorRtRtxExecutionProvider",
        other => bail!("unknown device '{other}'"),
    })
}

fn provider_is_available(name: &str) -> Result<bool> {
    match name {
        "CPUExecutionProvider" => Ok(true),
        "CUDAExecutionProvider" => ep::CUDA::default()
            .is_available()
            .map_err(|e| anyhow::anyhow!("{e}")),
        "DmlExecutionProvider" => ep::DirectML::default()
            .is_available()
            .map_err(|e| anyhow::anyhow!("{e}")),
        "OpenVINOExecutionProvider" => ep::OpenVINO::default()
            .is_available()
            .map_err(|e| anyhow::anyhow!("{e}")),
        name if windows_ml::is_catalog_provider_name(name) => {
            windows_ml::provider_is_registered(name)
        }
        other => bail!("unsupported execution provider '{other}'"),
    }
}

fn available_provider_summary() -> Result<String> {
    let mut out = Vec::new();
    for name in [
        "CPUExecutionProvider",
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "QNNExecutionProvider",
        "OpenVINOExecutionProvider",
        "VitisAIExecutionProvider",
        "MIGraphXExecutionProvider",
        "NvTensorRtRtxExecutionProvider",
    ] {
        if provider_is_available(name)? {
            out.push(name);
        }
    }
    Ok(out.join(", "))
}

fn is_windows_ml_group_alias(name: &str) -> bool {
    matches!(
        name.trim().to_ascii_lowercase().as_str(),
        "windowsml" | "winml" | "windows-ml" | "windows_ml"
    )
}

fn resolve_windows_ml_provider_names(ort_lib: Option<&Path>) -> Result<Vec<String>> {
    windows_ml::register_byo_provider_libraries()?;
    windows_ml::register_installed_certified(ort_lib)?;
    let registered = windows_ml::registered_provider_names()?;
    let mut out = Vec::new();
    for preferred in windows_ml::CATALOG_PROVIDER_ORDER {
        if registered
            .iter()
            .any(|name| windows_ml::provider_name_eq(name, preferred))
        {
            out.push((*preferred).to_string());
        }
    }
    if out.is_empty() {
        bail!(
            "no Windows ML catalog execution providers are installed and registered; use --device directml for legacy DirectML or install a Windows ML catalog EP for this device"
        );
    }
    Ok(out)
}
