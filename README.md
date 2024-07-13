# Study Program Text Classifier API

The Study Program Text Classifier API is a RESTful API that classifies indonesian abstract texts into study programs (Manajemen, Akuntansi, Teknik Informatika, Bahasa dan Sastra Inggris, Desain dan Komunikasi Visual). This API accepts text abstracts and returns the predicted study program with confidence score and it allows training the pretrained model on new data.

## Features

- Classify text abstracts into study programs and receive the prediction results with confidence score.
- Train the pretrained model to learn from new data (of course indonesian abstract text).

## Technologies

- Python
- PyTorch
- Gensim (Word2Vec)
- Pandas

## API Endpoints

### 1. Classify Text Abstract

- **URL:** `/api/predict`
- **Method:** `POST`
- **Headers:**
  ```http
  Content-Type: application/json
  ```
- **Body:**
  ```json
  {
      "abstracts": ["Your text abstract here", ...]
  }
  ```
- **Response:**
  ```json
  {
      "message": "success",
      "results": ...
  }
  ```

## Example Requests

### Classify Text Abstract

Here is an example using `curl` to classify a text abstract:

```bash
curl -X POST http://localhost:5000/api/classify -H "Content-Type: application/json" -d '{"text": "An abstract about advanced computing techniques"}'
```
