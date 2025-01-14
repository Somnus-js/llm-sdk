use serde::{Deserialize, Serialize};

use crate::IntoRequest;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatcompletionRequest {}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {}

impl IntoRequest for ChatcompletionRequest {
    fn into_request(self, client: reqwest::Client) -> reqwest::RequestBuilder {
        client.post("https://api.openai.com/v1/chat/completions")
            .json(&self)
    }
}
