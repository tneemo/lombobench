import unittest

import httpx

from lombo.huggingface import HuggingFaceHubService


class HuggingFaceHubServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_lists_models_with_filters_and_normalizes_metadata(self):
        async def handler(request):
            self.assertEqual(request.url.path, "/api/models")
            self.assertEqual(request.url.params["author"], "openai-community")
            self.assertEqual(request.url.params["pipeline_tag"], "text-generation")
            self.assertEqual(request.url.params["filter"], "license:mit")
            return httpx.Response(200, headers={
                "Link": '<https://huggingface.co/api/models?cursor=next-token>; rel="next"'
            }, json=[{
                "id": "openai-community/gpt2", "author": "openai-community",
                "downloads": 10, "likes": 3, "pipeline_tag": "text-generation",
                "lastModified": "2026-01-01T00:00:00Z", "gated": False,
                "tags": ["transformers", "license:mit"],
                "cardData": {"license": "mit"},
                "config": {"architectures": ["GPT2LMHeadModel"]},
            }])

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = HuggingFaceHubService(client)
            page = await service.list_models({
                "author": "openai-community", "pipeline": "text-generation",
                "license": "mit", "limit": 12,
            })

        self.assertEqual(page["items"][0]["architecture"], "GPT2LMHeadModel")
        self.assertEqual(page["items"][0]["license"], "mit")
        self.assertEqual(page["next_cursor"], "next-token")

    async def test_model_detail_combines_metadata_card_and_config(self):
        async def handler(request):
            if request.url.path == "/api/models/org/model":
                return httpx.Response(200, json={
                    "id": "org/model", "siblings": [{"rfilename": "weights.bin", "size": 42}],
                    "tags": [], "gated": "auto", "downloads": 1, "likes": 2,
                })
            if request.url.path == "/org/model/resolve/main/README.md":
                return httpx.Response(200, text="# Model card")
            if request.url.path == "/org/model/resolve/main/config.json":
                return httpx.Response(200, json={
                    "architectures": ["ExampleForCausalLM"], "max_position_embeddings": 8192,
                })
            return httpx.Response(404)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            detail = await HuggingFaceHubService(client).get_model("org/model")

        self.assertEqual(detail["card"], "# Model card")
        self.assertEqual(detail["context_length"], 8192)
        self.assertEqual(detail["architecture"], "ExampleForCausalLM")
        self.assertEqual(detail["files"][0]["size"], 42)

    async def test_dataset_detail_uses_viewer_splits_features_and_size(self):
        async def handler(request):
            if request.url.host == "datasets-server.huggingface.co":
                return httpx.Response(200, json={"dataset_info": {"default": {
                    "features": {"text": {"dtype": "string"}},
                    "splits": [{"name": "train", "num_examples": 5, "num_bytes": 100}],
                    "dataset_size": 100,
                }}})
            if request.url.path == "/api/datasets/org/data":
                return httpx.Response(200, json={
                    "id": "org/data", "author": "org", "siblings": [], "tags": ["modality:text"],
                    "downloads": 5, "likes": 1, "cardData": {"license": ["cc-by-4.0"]},
                })
            if request.url.path == "/datasets/org/data/resolve/main/README.md":
                return httpx.Response(200, text="# Dataset card")
            return httpx.Response(404)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            detail = await HuggingFaceHubService(client).get_dataset("org/data")

        self.assertEqual(detail["size"], 100)
        self.assertEqual(detail["splits"][0]["name"], "train")
        self.assertEqual(detail["features"]["default"]["text"]["dtype"], "string")
        self.assertEqual(detail["license"], "cc-by-4.0")


if __name__ == "__main__":
    unittest.main()
