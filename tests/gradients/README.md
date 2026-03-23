# Gradients (可扩展区)

当前留空，预留后续放置梯度回归与 End-to-End 可微优化测试。

建议目录结构：
- `tests/gradients/e2e/`：端到端梯度流程（场景构建 -> Simulation -> Result -> 回传）
- `tests/gradients/unit/`：单元级梯度核验（编译梯度、监视器梯度、postprocess梯度）

示例测试命名：
- `test_gradient_scene_to_result_e2e.py`
- `test_gradient_s_parameters.py`
