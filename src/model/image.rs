use std::collections::BTreeMap;

use burn::prelude::*;

use image::{GenericImageView, Pixel};
use itertools::Itertools;
use rayon::iter::{ParallelBridge, ParallelIterator};

#[derive(Clone)]
pub struct Image<B: Backend> {
    tensor: Tensor<B, 3, burn::tensor::Int>,
}

impl<B: Backend> Image<B> {
    pub fn new<I>(image: &I, device: &B::Device) -> Result<Self, burn::config::ConfigError>
    where
        I: image::GenericImageView + Send + Sync,
        I::Pixel: Pixel<Subpixel = u8>,
    {
        let ch_count = I::Pixel::CHANNEL_COUNT as usize;

        let shape = Shape::new([image.height() as usize, image.width() as usize, ch_count]);

        let pixels = (0..image.height())
            .par_bridge()
            .map(|row_idx| {
                let row = image.view(0, row_idx, image.width(), 1);
                let row = row.pixels().fold(
                    Vec::with_capacity(image.width() as usize * ch_count),
                    |mut vec, (_, _, p)| {
                        vec.extend(p.channels().iter().copied().map(i32::from));
                        vec
                    },
                );

                assert_eq!(row.len(), image.width() as usize * ch_count);

                let data = Data::new(row, Shape::new([image.width() as usize, ch_count]));
                let t = Tensor::from_ints(data, device);
                (row_idx, t)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .sorted_by_key(|(k, _)| *k)
            .map(|(_, t)| t)
            .collect::<Vec<_>>();

        let tensor = Tensor::stack(pixels, 0);

        assert_eq!(shape, tensor.shape());

        Ok(Self { tensor })
    }

    pub fn height(&self) -> usize {
        self.tensor.shape().dims[0]
    }

    pub fn width(&self) -> usize {
        self.tensor.shape().dims[1]
    }

    pub fn channels(&self) -> usize {
        self.tensor.shape().dims[2]
    }

    pub fn into_image(self) -> image::DynamicImage {
        let data = self.tensor.into_data().convert::<u8>();

        match data.shape.dims[2] {
            1 => image::DynamicImage::ImageLuma8(
                image::GrayImage::from_vec(
                    data.shape.dims[1] as u32,
                    data.shape.dims[0] as u32,
                    data.value,
                )
                .unwrap(),
            ),
            2 => image::DynamicImage::ImageLumaA8(
                image::GrayAlphaImage::from_vec(
                    data.shape.dims[1] as u32,
                    data.shape.dims[0] as u32,
                    data.value,
                )
                .unwrap(),
            ),
            3 => image::DynamicImage::ImageRgb8(
                image::RgbImage::from_vec(
                    data.shape.dims[1] as u32,
                    data.shape.dims[0] as u32,
                    data.value,
                )
                .unwrap(),
            ),
            4 => image::DynamicImage::ImageRgba8(
                image::RgbaImage::from_vec(
                    data.shape.dims[1] as u32,
                    data.shape.dims[0] as u32,
                    data.value,
                )
                .unwrap(),
            ),
            ch => panic!("Invalid number of channels: {ch}"),
        }
    }
}

#[test]
fn test() {
    let f = std::fs::File::open("image001.png").unwrap();
    let image = image::load(std::io::BufReader::new(f), image::ImageFormat::Png).unwrap();
    let image = image::imageops::colorops::grayscale_alpha(&image);
    let image_cmp = image::DynamicImage::ImageLumaA8(image.clone());
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let image_burn = Image::<burn_ndarray::NdArray>::new(&image, &device).unwrap();
    let image_export = image_burn.clone().into_image();

    image_export.save("image001-out.png").unwrap();

    println!(
        "{} x {} x {}",
        image_burn.height(),
        image_burn.width(),
        image_burn.channels()
    );

    assert!(image_export == image_cmp, "Images are different");
}
