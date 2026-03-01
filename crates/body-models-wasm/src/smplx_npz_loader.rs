//! Load SMPL-X model directly from .npz files.

use std::io::{Cursor, Read};

use crate::smplx_model_data::SmplxModelData;

#[derive(Debug)]
pub enum NpzError {
    Zip(String),
    Npy(String),
    MissingArray(&'static str),
    ShapeMismatch { name: &'static str, expected: &'static str, got: String },
}

impl std::fmt::Display for NpzError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NpzError::Zip(e) => write!(f, "ZIP error: {}", e),
            NpzError::Npy(e) => write!(f, "NPY parse error: {}", e),
            NpzError::MissingArray(name) => write!(f, "Missing array '{}' in npz", name),
            NpzError::ShapeMismatch { name, expected, got } => {
                write!(f, "Array '{}' shape mismatch: expected {}, got {}", name, expected, got)
            }
        }
    }
}

impl std::error::Error for NpzError {}

struct NpyArray {
    shape: Vec<usize>,
    dtype: String,
    fortran_order: bool,
    data: Vec<u8>,
}

fn parse_npy<R: Read>(reader: &mut R) -> Result<NpyArray, NpzError> {
    let mut magic = [0u8; 6];
    reader.read_exact(&mut magic).map_err(|e| NpzError::Npy(format!("read magic: {}", e)))?;
    if &magic[..6] != b"\x93NUMPY" {
        return Err(NpzError::Npy("bad magic".into()));
    }

    let mut ver = [0u8; 2];
    reader.read_exact(&mut ver).map_err(|e| NpzError::Npy(format!("read version: {}", e)))?;
    let major = ver[0];

    let header_len = if major <= 1 {
        let mut buf = [0u8; 2];
        reader.read_exact(&mut buf).map_err(|e| NpzError::Npy(format!("read header len: {}", e)))?;
        u16::from_le_bytes(buf) as usize
    } else {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf).map_err(|e| NpzError::Npy(format!("read header len v2: {}", e)))?;
        u32::from_le_bytes(buf) as usize
    };

    let mut header_bytes = vec![0u8; header_len];
    reader.read_exact(&mut header_bytes).map_err(|e| NpzError::Npy(format!("read header: {}", e)))?;
    let header = String::from_utf8_lossy(&header_bytes);

    let dtype = extract_field(&header, "descr").ok_or_else(|| NpzError::Npy("missing 'descr'".into()))?;
    let fortran_order = extract_field(&header, "fortran_order")
        .unwrap_or_else(|| "False".into())
        .contains("True");
    let shape = extract_shape(&header).ok_or_else(|| NpzError::Npy("missing 'shape'".into()))?;

    let num_elements: usize = shape.iter().product::<usize>().max(1);
    let element_size = dtype_size(&dtype)?;
    let data_size = num_elements * element_size;
    let mut data = vec![0u8; data_size];
    reader.read_exact(&mut data).map_err(|e| NpzError::Npy(format!("read data: {}", e)))?;

    Ok(NpyArray { shape, dtype, fortran_order, data })
}

fn extract_field(header: &str, field: &str) -> Option<String> {
    let pattern = format!("'{}':", field);
    let start = header.find(&pattern)? + pattern.len();
    let rest = header[start..].trim_start();
    if rest.starts_with('\'') || rest.starts_with('"') {
        let quote = rest.as_bytes()[0] as char;
        let end = rest[1..].find(quote)? + 1;
        Some(rest[1..end].to_string())
    } else {
        let end = rest.find([',', '}'].as_ref()).unwrap_or(rest.len());
        Some(rest[..end].trim().to_string())
    }
}

fn extract_shape(header: &str) -> Option<Vec<usize>> {
    let start = header.find("'shape':")? + "'shape':".len();
    let rest = header[start..].trim_start();
    let paren_start = rest.find('(')?;
    let paren_end = rest.find(')')?;
    let inner = &rest[paren_start + 1..paren_end];
    if inner.trim().is_empty() {
        return Some(vec![]);
    }
    let dims: Result<Vec<usize>, _> = inner
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>())
        .collect();
    dims.ok()
}

fn dtype_size(dtype: &str) -> Result<usize, NpzError> {
    let clean = dtype.trim_start_matches(['<', '>', '|', '=']);
    match clean.chars().next() {
        Some('f') | Some('u') | Some('i') => {
            let n: usize = clean[1..].parse().map_err(|_| NpzError::Npy(format!("bad dtype: {}", dtype)))?;
            Ok(n)
        }
        Some('U') => {
            let n: usize = clean[1..].parse().map_err(|_| NpzError::Npy(format!("bad dtype: {}", dtype)))?;
            Ok(n * 4)
        }
        Some('b') => Ok(1),
        _ => Err(NpzError::Npy(format!("unsupported dtype: {}", dtype))),
    }
}

impl NpyArray {
    fn to_f32(&self) -> Result<Vec<f32>, NpzError> {
        if self.fortran_order && self.shape.len() > 1 {
            return self.to_f32_transpose();
        }
        let n: usize = self.shape.iter().product::<usize>().max(1);
        let clean_dtype = self.dtype.trim_start_matches(['<', '>', '|', '=']);
        match clean_dtype {
            "f8" => {
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    let bytes: [u8; 8] = self.data[i * 8..(i + 1) * 8].try_into().unwrap();
                    out.push(f64::from_le_bytes(bytes) as f32);
                }
                Ok(out)
            }
            "f4" => {
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    let bytes: [u8; 4] = self.data[i * 4..(i + 1) * 4].try_into().unwrap();
                    out.push(f32::from_le_bytes(bytes));
                }
                Ok(out)
            }
            _ => Err(NpzError::Npy(format!("cannot convert dtype '{}' to f32", self.dtype))),
        }
    }

    fn to_i32(&self) -> Result<Vec<i32>, NpzError> {
        let n: usize = self.shape.iter().product::<usize>().max(1);
        let clean_dtype = self.dtype.trim_start_matches(['<', '>', '|', '=']);
        match clean_dtype {
            "i4" => {
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    let bytes: [u8; 4] = self.data[i * 4..(i + 1) * 4].try_into().unwrap();
                    out.push(i32::from_le_bytes(bytes));
                }
                Ok(out)
            }
            "u4" => {
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    let bytes: [u8; 4] = self.data[i * 4..(i + 1) * 4].try_into().unwrap();
                    out.push(u32::from_le_bytes(bytes) as i32);
                }
                Ok(out)
            }
            _ => Err(NpzError::Npy(format!("cannot convert dtype '{}' to i32", self.dtype))),
        }
    }

    fn to_u32(&self) -> Result<Vec<u32>, NpzError> {
        let n: usize = self.shape.iter().product::<usize>().max(1);
        let clean_dtype = self.dtype.trim_start_matches(['<', '>', '|', '=']);
        match clean_dtype {
            "u4" => {
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    let bytes: [u8; 4] = self.data[i * 4..(i + 1) * 4].try_into().unwrap();
                    out.push(u32::from_le_bytes(bytes));
                }
                Ok(out)
            }
            "i4" => {
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    let bytes: [u8; 4] = self.data[i * 4..(i + 1) * 4].try_into().unwrap();
                    out.push(i32::from_le_bytes(bytes) as u32);
                }
                Ok(out)
            }
            _ => Err(NpzError::Npy(format!("cannot convert dtype '{}' to u32", self.dtype))),
        }
    }

    fn to_f32_transpose(&self) -> Result<Vec<f32>, NpzError> {
        let shape = &self.shape;
        let ndim = shape.len();
        let n: usize = shape.iter().product();
        let clean_dtype = self.dtype.trim_start_matches(['<', '>', '|', '=']);

        let raw: Vec<f64> = match clean_dtype {
            "f8" => (0..n)
                .map(|i| {
                    let bytes: [u8; 8] = self.data[i * 8..(i + 1) * 8].try_into().unwrap();
                    f64::from_le_bytes(bytes)
                })
                .collect(),
            "f4" => (0..n)
                .map(|i| {
                    let bytes: [u8; 4] = self.data[i * 4..(i + 1) * 4].try_into().unwrap();
                    f32::from_le_bytes(bytes) as f64
                })
                .collect(),
            _ => return Err(NpzError::Npy(format!("cannot convert Fortran dtype '{}' to f32", self.dtype))),
        };

        let mut f_strides = vec![1usize; ndim];
        for i in 1..ndim {
            f_strides[i] = f_strides[i - 1] * shape[i - 1];
        }
        let mut c_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            c_strides[i] = c_strides[i + 1] * shape[i + 1];
        }

        let mut out = vec![0.0f32; n];
        let mut indices = vec![0usize; ndim];
        for c_idx in 0..n {
            let mut rem = c_idx;
            for d in 0..ndim {
                indices[d] = rem / c_strides[d];
                rem %= c_strides[d];
            }
            let f_idx: usize = indices.iter().zip(f_strides.iter()).map(|(&i, &s)| i * s).sum();
            out[c_idx] = raw[f_idx] as f32;
        }
        Ok(out)
    }
}

fn read_npy_from_zip(
    archive: &mut zip::ZipArchive<Cursor<&[u8]>>,
    name: &'static str,
) -> Result<NpyArray, NpzError> {
    let file_name = format!("{}.npy", name);
    let has_npy = archive.by_name(&file_name).is_ok();
    let actual_name = if has_npy { file_name.as_str() } else { name };
    let mut file = archive.by_name(actual_name).map_err(|_| NpzError::MissingArray(name))?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).map_err(|e| NpzError::Zip(format!("read {}: {}", name, e)))?;
    let mut cursor = Cursor::new(buf.as_slice());
    parse_npy(&mut cursor)
}

/// Load an SMPL-X model from `.npz` bytes.
pub fn load_npz(data: &[u8]) -> Result<SmplxModelData, NpzError> {
    let cursor = Cursor::new(data);
    let mut archive = zip::ZipArchive::new(cursor).map_err(|e| NpzError::Zip(format!("{}", e)))?;

    let v_template_arr = read_npy_from_zip(&mut archive, "v_template")?;
    let shapedirs_arr = read_npy_from_zip(&mut archive, "shapedirs")?;
    let posedirs_arr = read_npy_from_zip(&mut archive, "posedirs")?;
    let j_regressor_arr = read_npy_from_zip(&mut archive, "J_regressor")?;
    let weights_arr = read_npy_from_zip(&mut archive, "weights")?;
    let faces_arr = read_npy_from_zip(&mut archive, "f")?;
    let kintree_arr = read_npy_from_zip(&mut archive, "kintree_table")?;
    let hands_mean_l_arr = read_npy_from_zip(&mut archive, "hands_meanl")?;
    let hands_mean_r_arr = read_npy_from_zip(&mut archive, "hands_meanr")?;

    if v_template_arr.shape.len() != 2 || v_template_arr.shape[1] != 3 {
        return Err(NpzError::ShapeMismatch {
            name: "v_template",
            expected: "[V,3]",
            got: format!("{:?}", v_template_arr.shape),
        });
    }
    if shapedirs_arr.shape.len() != 3 || shapedirs_arr.shape[0] != v_template_arr.shape[0] || shapedirs_arr.shape[1] != 3 {
        return Err(NpzError::ShapeMismatch {
            name: "shapedirs",
            expected: "[V,3,S_total]",
            got: format!("{:?}", shapedirs_arr.shape),
        });
    }
    if j_regressor_arr.shape.len() != 2 || j_regressor_arr.shape[1] != v_template_arr.shape[0] {
        return Err(NpzError::ShapeMismatch {
            name: "J_regressor",
            expected: "[J,V]",
            got: format!("{:?}", j_regressor_arr.shape),
        });
    }
    if weights_arr.shape.len() != 2 || weights_arr.shape[0] != v_template_arr.shape[0] {
        return Err(NpzError::ShapeMismatch {
            name: "weights",
            expected: "[V,J]",
            got: format!("{:?}", weights_arr.shape),
        });
    }
    if kintree_arr.shape.len() != 2 || kintree_arr.shape[0] < 1 {
        return Err(NpzError::ShapeMismatch {
            name: "kintree_table",
            expected: "[2,J]",
            got: format!("{:?}", kintree_arr.shape),
        });
    }
    if hands_mean_l_arr.shape.iter().product::<usize>() != 45 {
        return Err(NpzError::ShapeMismatch {
            name: "hands_meanl",
            expected: "[45]",
            got: format!("{:?}", hands_mean_l_arr.shape),
        });
    }
    if hands_mean_r_arr.shape.iter().product::<usize>() != 45 {
        return Err(NpzError::ShapeMismatch {
            name: "hands_meanr",
            expected: "[45]",
            got: format!("{:?}", hands_mean_r_arr.shape),
        });
    }

    let v = v_template_arr.shape[0];
    let j = j_regressor_arr.shape[0];
    let total_dirs = shapedirs_arr.shape[2];
    let s = total_dirs.min(300);
    let e = total_dirs.saturating_sub(300);

    let v_template = v_template_arr.to_f32()?;
    let shapedirs_full = shapedirs_arr.to_f32()?;
    let j_regressor = j_regressor_arr.to_f32()?;
    let lbs_weights = weights_arr.to_f32()?;
    let faces = faces_arr.to_u32()?;
    let kintree = kintree_arr.to_i32()?;
    let hand_mean_l = hands_mean_l_arr.to_f32()?;
    let hand_mean_r = hands_mean_r_arr.to_f32()?;

    let mut hand_mean = vec![0.0f32; 90];
    hand_mean[..45].copy_from_slice(&hand_mean_l[..45]);
    hand_mean[45..].copy_from_slice(&hand_mean_r[..45]);

    let mut shapedirs = vec![0.0f32; v * 3 * s];
    for vi in 0..v {
        for di in 0..3 {
            for si in 0..s {
                let src = vi * 3 * total_dirs + di * total_dirs + si;
                shapedirs[vi * 3 * s + di * s + si] = shapedirs_full[src];
            }
        }
    }
    let mut exprdirs = vec![0.0f32; v * 3 * e];
    for vi in 0..v {
        for di in 0..3 {
            for ei in 0..e {
                let src = vi * 3 * total_dirs + di * total_dirs + (300 + ei);
                exprdirs[vi * 3 * e + di * e + ei] = shapedirs_full[src];
            }
        }
    }

    // posedirs [P, V*3]
    let p = if posedirs_arr.shape.len() == 2 {
        posedirs_arr.shape[0]
    } else if posedirs_arr.shape.len() == 3 {
        posedirs_arr.shape[2]
    } else {
        return Err(NpzError::ShapeMismatch {
            name: "posedirs",
            expected: "[P,V*3] or [V,3,P]",
            got: format!("{:?}", posedirs_arr.shape),
        });
    };
    let posedirs = if posedirs_arr.shape.len() == 2 {
        let rows = posedirs_arr.shape[0];
        let cols = posedirs_arr.shape[1];
        if cols != v * 3 {
            return Err(NpzError::ShapeMismatch {
                name: "posedirs",
                expected: "[P,V*3]",
                got: format!("{:?}", posedirs_arr.shape),
            });
        }
        let raw = posedirs_arr.to_f32()?;
        let mut out = vec![0.0f32; rows * cols];
        out.copy_from_slice(&raw);
        out
    } else {
        // [V,3,P] -> [P,V*3]
        let raw = posedirs_arr.to_f32()?;
        let mut out = vec![0.0f32; p * v * 3];
        for vi in 0..v {
            for di in 0..3 {
                let vid = vi * 3 + di;
                for pi in 0..p {
                    out[pi * v * 3 + vid] = raw[vi * 3 * p + di * p + pi];
                }
            }
        }
        out
    };

    let mut parents = vec![0i32; j];
    for ji in 0..j {
        parents[ji] = kintree[ji];
    }
    if parents[0] != -1 {
        parents[0] = -1;
    }

    let mut j_template = vec![0.0f32; j * 3];
    for ji in 0..j {
        for d in 0..3 {
            let mut sum = 0.0f32;
            for vi in 0..v {
                sum += j_regressor[ji * v + vi] * v_template[vi * 3 + d];
            }
            j_template[ji * 3 + d] = sum;
        }
    }

    let mut j_shapedirs = vec![0.0f32; j * 3 * s];
    for ji in 0..j {
        for d in 0..3 {
            for si in 0..s {
                let mut sum = 0.0f32;
                for vi in 0..v {
                    sum += j_regressor[ji * v + vi] * shapedirs[vi * 3 * s + d * s + si];
                }
                j_shapedirs[ji * 3 * s + d * s + si] = sum;
            }
        }
    }

    let mut j_exprdirs = vec![0.0f32; j * 3 * e];
    for ji in 0..j {
        for d in 0..3 {
            for ei in 0..e {
                let mut sum = 0.0f32;
                for vi in 0..v {
                    sum += j_regressor[ji * v + vi] * exprdirs[vi * 3 * e + d * e + ei];
                }
                j_exprdirs[ji * 3 * e + d * e + ei] = sum;
            }
        }
    }

    let mut min_y = f32::INFINITY;
    for vi in 0..v {
        min_y = min_y.min(v_template[vi * 3 + 1]);
    }
    let rest_pose_y_offset = -min_y;

    Ok(SmplxModelData {
        num_vertices: v,
        num_joints: j,
        num_shape_params: s,
        num_expr_params: e,
        num_pose_params: p,
        rest_pose_y_offset,
        v_template,
        shapedirs,
        exprdirs,
        posedirs,
        j_template,
        j_shapedirs,
        j_exprdirs,
        parents,
        lbs_weights,
        faces,
        hand_mean,
    })
}
