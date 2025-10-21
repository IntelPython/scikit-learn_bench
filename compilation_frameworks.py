import numpy as np
import onnx
import onnxruntime as ort
import tvm
from tvm import relay
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestClassifier
from time import time

# Generate sample data
X_train = np.random.rand(100, 10).astype(np.float32)
y_train = np.random.randint(0, 2, size=(100,))

# Train a simple RandomForest model
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)

# Convert model to ONNX
initial_type = [("input", FloatTensorType([None, 10]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)
onnx.save_model(onnx_model, "model.onnx")

# Load ONNX model for inference test
ort_session = ort.InferenceSession("model.onnx")
input_data = {ort_session.get_inputs()[0].name: X_train[:5]}
start = time()
ort_outs = ort_session.run(None, input_data)
print(f"ONNX Inference Time: {time() - start:.4f}s")

# Optimize ONNX model with TVM
onnx_model = onnx.load("model.onnx")
mod, params = relay.frontend.from_onnx(onnx_model, shape={"input": (1, 10)})

# Compile with TVM
target = "llvm"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

# Run inference with TVM
dev = tvm.cpu()
dtype = "float32"
tvm_model = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
tvm_model.set_input("input", tvm.nd.array(X_train[:5].astype(dtype)))

start = time()
tvm_model.run()
tvm_out = tvm_model.get_output(0).numpy()
print(f"TVM Optimized Inference Time: {time() - start:.4f}s")

print("Optimization complete! Compare ONNX vs. TVM inference times.")
