use derive_builder::Builder;
use reqwest::RequestBuilder;
use serde::{Deserialize, Serialize};

use crate::IntoRequest;

#[derive(Debug, Clone, Serialize, Builder)]
#[builder(pattern = "mutable")]
pub struct CreateImageRequest {
    #[builder(setter(into))]
    prompt: String,
    #[builder(default)]
    model: ImageModel,
    /// The number of images to generate   
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<usize>,
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    quality: Option<ImageQuality>,
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ImageResposeFormat>,
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<ImageSize>,
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    style: Option<ImageStyle>,
    #[builder(default, setter(strip_option, into))]
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageModel {
    #[serde(rename = "dall-e-3")]
    DallE3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageQuality {
    Standard,
    Hd,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageResposeFormat {
    Url,
    B64Json,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageSize {
    #[serde(rename = "1024x1024")]
    Large,
    #[serde(rename = "1792x1024")]
    LargeWide,
    #[serde(rename = "1024*1792")]
    LargeTall,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageStyle {
    Vivid,
    Natural,
}
#[derive(Debug, Clone, Deserialize)]
pub struct CreateImageResponse {
    pub created: u64,
    pub data: Vec<ImageObject>,
}
#[derive(Debug, Clone, Deserialize)]
pub struct ImageObject {
    pub b64_json: Option<String>,
    pub url: Option<String>,
    pub revised_prompt: String,
}
impl IntoRequest for CreateImageRequest {
    fn into_request(self, client: reqwest::Client) -> RequestBuilder {
        client
            .post("https://api.openai.com/v1/images/generations")
            .json(&self)
    }
}
impl CreateImageRequest {
    pub fn new(prompt: impl Into<String>) -> Self {
        CreateImageRequestBuilder::default()
            .prompt(prompt)
            .build()
            .unwrap()
    }
}
impl Default for ImageModel {
    fn default() -> Self {
        ImageModel::DallE3
    }
}
impl Default for ImageQuality {
    fn default() -> Self {
        ImageQuality::Standard
    }
}
impl Default for ImageResposeFormat {
    fn default() -> Self {
        ImageResposeFormat::Url
    }
}
impl Default for ImageSize {
    fn default() -> Self {
        ImageSize::Large
    }
}
impl Default for ImageStyle {
    fn default() -> Self {
        ImageStyle::Vivid
    }
}

#[cfg(test)]
mod test {
    use std::fs;

    use crate::LlmSdk;

    use super::*;
    use anyhow::{Ok, Result};
    use serde_json::json;
    #[test]
    fn create_iamge_request_should_serialize() -> Result<()> {
        let req = CreateImageRequest::new("draw a cute cat");
        assert_eq!(
            serde_json::to_value(&req)?,
            json!({
                "prompt": "draw a cute cat",
                "model": "dall-e-3",
            })
        );
        Ok(())
    }
    #[test]
    fn create_iamge_request_custom_should_serialize() -> Result<()> {
        let req = CreateImageRequestBuilder::default()
            .prompt("draw a cute caterpillar")
            .style(ImageStyle::Natural)
            .quality(ImageQuality::Hd)
            .build()?;
        assert_eq!(
            serde_json::to_value(&req)?,
            json!({
                "prompt": "draw a cute caterpillar",
                "model": "dall-e-3",
                "style": "natural",
                "quality": "hd"
            })
        );
        Ok(())
    }
    #[tokio::test]
    async fn create_image_should_work() -> Result<()> {
        let sdk = LlmSdk::new(std::env::var("OPENAI_API_KEY")?);
        let req = CreateImageRequest::new("draw a cute caterpillar");
        let res = sdk.create_image(req).await?;
        assert_eq!(res.data.len(), 1);
        let image = &res.data[0];
        assert!(image.url.is_some());
        assert!(image.b64_json.is_none());
        fs::write(
            "/tmp/caterpillar.png",
            reqwest::get(image.url.as_ref().unwrap())
                .await?
                .bytes()
                .await?,
        )?;
        Ok(())
    }
}
