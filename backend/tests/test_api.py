"""
FastAPI 接口集成测试（使用 httpx + TestClient，不启动 Celery）。
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from api.main import app
from unittest.mock import patch

client = TestClient(app)

def _model_ir_payload():
    return {
        "id": "model-api-001",
        "name": "Test Cnn",
        "version": "1.0.0",
        "nodes": [
            {"id": "n0", "op_type": "Input",   "name": "input",   "params": {"shape": [3, 32, 32]}},
            {"id": "n1", "op_type": "Conv2d",  "name": "conv1",   "params": {"in_channels": 3, "out_channels": 16, "kernel_size": 3, "padding": 1}},
            {"id": "n2", "op_type": "Flatten", "name": "flatten", "params": {}},
            {"id": "n3", "op_type": "Linear",  "name": "fc1",     "params": {"in_features": 16384, "out_features": 10}},
            {"id": "n4", "op_type": "Output",  "name": "output",  "params": {}},
        ],
        "edges": [
            {"id": "e01", "source": "n0", "target": "n1"},
            {"id": "e12", "source": "n1", "target": "n2"},
            {"id": "e23", "source": "n2", "target": "n3"},
            {"id": "e34", "source": "n3", "target": "n4"},
        ],
    }

def _data_ir_payload():
    return {
        "id": "data-api-001",
        "name": "TestData",
        "source": {"type": "torchvision", "dataset_name": "CIFAR10", "download_root": "./data"},
        "schema": {"task_type": "image_classification", "num_classes": 10},
        "split": {"strategy": "ratio", "train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1},
        "train_pipeline": {"transforms": []},
        "val_pipeline": {"transforms": []},
    }

def _experiment_ir_payload():
    return {
        "id": "exp-api-001",
        "name": "TestExperiment",
        "model_ir_id": "model-api-001",
        "data_ir_id":  "data-api-001",
        "hyper_params": {"epochs": 1},
        "backend": {"type": "local", "device": "cpu"},
        "checkpoint": {"enabled": False},
    }

# ── 测试 Model IR ──

def test_create_and_get_model_ir():
    resp = client.post("/api/v1/model-irs", json=_model_ir_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["id"] == "model-api-001"
    print("✅ POST /model-irs 通过")

    resp = client.get("/api/v1/model-irs/model-api-001")
    assert resp.status_code == 200
    assert resp.json()["data"]["name"] == "Test Cnn"
    print("✅ GET /model-irs/{id} 通过")

def test_codegen_endpoint():
    client.post("/api/v1/model-irs", json=_model_ir_payload())
    resp = client.get("/api/v1/model-irs/model-api-001/codegen")
    assert resp.status_code == 200
    assert "class TestCnn(nn.Module)" in resp.text
    print("✅ GET /model-irs/{id}/codegen 通过")

def test_delete_model_ir():
    client.post("/api/v1/model-irs", json=_model_ir_payload())
    resp = client.delete("/api/v1/model-irs/model-api-001")
    assert resp.status_code == 200
    resp = client.get("/api/v1/model-irs/model-api-001")
    assert resp.status_code == 404
    print("✅ DELETE /model-irs/{id} 通过")

# ── 测试 Data IR ──

def test_create_and_list_data_ir():
    resp = client.post("/api/v1/data-irs", json=_data_ir_payload())
    assert resp.status_code == 200
    assert resp.json()["data"]["num_classes"] == 10
    print("✅ POST /data-irs 通过")

    resp = client.get("/api/v1/data-irs")
    assert resp.status_code == 200
    assert len(resp.json()["data"]) >= 1
    print("✅ GET /data-irs 通过")

# ── 测试 Experiment ──

def test_create_experiment():
    client.post("/api/v1/model-irs", json=_model_ir_payload())
    client.post("/api/v1/data-irs",  json=_data_ir_payload())
    resp = client.post("/api/v1/experiments", json=_experiment_ir_payload())
    assert resp.status_code == 200
    assert resp.json()["data"]["id"] == "exp-api-001"
    print("✅ POST /experiments 通过")

def test_submit_training_mocked():
    """Mock Celery task，避免真实训练"""
    client.post("/api/v1/model-irs",  json=_model_ir_payload())
    client.post("/api/v1/data-irs",   json=_data_ir_payload())
    client.post("/api/v1/experiments", json=_experiment_ir_payload())

    with patch("api.routers.experiment.run_training") as mock_task:
        mock_task.delay.return_value.id = "mock-celery-task-001"
        resp = client.post("/api/v1/experiments/exp-api-001/submit")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["celery_task_id"] == "mock-celery-task-001"
        print("✅ POST /experiments/{id}/submit 通过（Mock Celery）")

if __name__ == "__main__":
    test_create_and_get_model_ir()
    test_codegen_endpoint()
    test_delete_model_ir()
    test_create_and_list_data_ir()
    test_create_experiment()
    test_submit_training_mocked()
    print("\n🎉 所有 API 测试通过")