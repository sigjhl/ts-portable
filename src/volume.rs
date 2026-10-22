use anyhow::{Context, Result, bail};

#[derive(Clone, Debug)]
pub struct Volume<T> {
    pub data: Vec<T>,
    pub shape: [usize; 3], // z, y, x
}

impl<T: Clone + Default> Volume<T> {
    pub fn zeros(shape: [usize; 3]) -> Self {
        let n = shape[0] * shape[1] * shape[2];
        Self {
            data: vec![T::default(); n],
            shape,
        }
    }
}

impl<T> Volume<T> {
    pub fn from_vec(shape: [usize; 3], data: Vec<T>) -> Result<Self> {
        let expected = shape[0] * shape[1] * shape[2];
        if data.len() != expected {
            bail!(
                "volume length mismatch: got {}, expected {} for shape {:?}",
                data.len(),
                expected,
                shape
            );
        }
        Ok(Self { data, shape })
    }

    #[inline]
    pub fn idx(&self, z: usize, y: usize, x: usize) -> usize {
        (z * self.shape[1] + y) * self.shape[2] + x
    }

    #[inline]
    pub fn get(&self, z: usize, y: usize, x: usize) -> &T {
        &self.data[self.idx(z, y, x)]
    }

    #[inline]
    pub fn get_mut(&mut self, z: usize, y: usize, x: usize) -> &mut T {
        let i = self.idx(z, y, x);
        &mut self.data[i]
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn map<U, F>(&self, mut f: F) -> Volume<U>
    where
        F: FnMut(&T) -> U,
    {
        Volume {
            data: self.data.iter().map(&mut f).collect(),
            shape: self.shape,
        }
    }
}

impl<T: Clone> Volume<T> {
    pub fn crop(&self, bbox: &[[usize; 2]; 3]) -> Result<Self> {
        let shape = [
            bbox[0][1].saturating_sub(bbox[0][0]),
            bbox[1][1].saturating_sub(bbox[1][0]),
            bbox[2][1].saturating_sub(bbox[2][0]),
        ];
        let mut out = Vec::with_capacity(shape[0] * shape[1] * shape[2]);
        for z in bbox[0][0]..bbox[0][1] {
            for y in bbox[1][0]..bbox[1][1] {
                let start = self.idx(z, y, bbox[2][0]);
                let end = self.idx(z, y, bbox[2][1] - 1) + 1;
                out.extend_from_slice(&self.data[start..end]);
            }
        }
        Volume::from_vec(shape, out).context("failed to build cropped volume")
    }

    pub fn paste(&mut self, crop: &Volume<T>, bbox: &[[usize; 2]; 3]) {
        for z in 0..crop.shape[0] {
            for y in 0..crop.shape[1] {
                for x in 0..crop.shape[2] {
                    let dz = bbox[0][0] + z;
                    let dy = bbox[1][0] + y;
                    let dx = bbox[2][0] + x;
                    if dz < self.shape[0] && dy < self.shape[1] && dx < self.shape[2] {
                        *self.get_mut(dz, dy, dx) = crop.get(z, y, x).clone();
                    }
                }
            }
        }
    }
}

pub fn product(shape: [usize; 3]) -> usize {
    shape[0] * shape[1] * shape[2]
}
