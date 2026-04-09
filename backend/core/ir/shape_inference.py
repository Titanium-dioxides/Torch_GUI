"""Shape Inference for Computation Graph."""

from typing import Dict, List, Optional, Tuple, Any
from .model_ir import ModelIR, Node, TensorShape, DataType


class ShapeInferenceError(Exception):
    """Exception raised when shape inference fails."""
    pass


def infer_binary_op_shape(input1: TensorShape, input2: TensorShape) -> TensorShape:
    """
    Infer output shape for binary operations with broadcasting.
    
    Supports NumPy-style broadcasting rules.
    """
    dims1 = input1.dims
    dims2 = input2.dims
    
    # Handle scalar case
    if not dims1:
        return TensorShape(dims=dims2.copy(), dtype=input2.dtype)
    if not dims2:
        return TensorShape(dims=dims1.copy(), dtype=input1.dtype)
    
    # Broadcasting
    result_dims = []
    for d1, d2 in zip(reversed(dims1), reversed(dims2)):
        if d1 == d2:
            result_dims.append(d1)
        elif d1 == 1:
            result_dims.append(d2)
        elif d2 == 1:
            result_dims.append(d1)
        else:
            raise ShapeInferenceError(
                f"Cannot broadcast shapes {dims1} and {dims2}"
            )
    
    # Add remaining dimensions from longer shape
    if len(dims1) > len(dims2):
        result_dims.extend(reversed(dims1[:len(dims1) - len(dims2)]))
    elif len(dims2) > len(dims1):
        result_dims.extend(reversed(dims2[:len(dims2) - len(dims1)]))
    
    result_dims.reverse()
    
    # Use higher precision dtype
    dtype = input1.dtype if input1.dtype.value >= input2.dtype.value else input2.dtype
    
    return TensorShape(dims=result_dims, dtype=dtype)


def infer_matmul_shape(input1: TensorShape, input2: TensorShape) -> TensorShape:
    """Infer output shape for matrix multiplication."""
    dims1 = input1.dims
    dims2 = input2.dims
    
    if len(dims1) < 2 or len(dims2) < 2:
        raise ShapeInferenceError("MatMul requires at least 2D tensors")
    
    if dims1[-1] != dims2[-2]:
        raise ShapeInferenceError(
            f"MatMul shape mismatch: {dims1} x {dims2}"
        )
    
    # Output shape: [..., M, N] where input1 is [..., M, K] and input2 is [..., K, N]
    result_dims = []
    
    # Handle batch dimensions with broadcasting
    batch_dims1 = dims1[:-2] if len(dims1) > 2 else []
    batch_dims2 = dims2[:-2] if len(dims2) > 2 else []
    
    if batch_dims1 and batch_dims2:
        # Broadcast batch dimensions
        for d1, d2 in zip(reversed(batch_dims1), reversed(batch_dims2)):
            if d1 == d2:
                result_dims.insert(0, d1)
            elif d1 == 1:
                result_dims.insert(0, d2)
            elif d2 == 1:
                result_dims.insert(0, d1)
            else:
                raise ShapeInferenceError(
                    f"Cannot broadcast batch dimensions {batch_dims1} and {batch_dims2}"
                )
        # Add remaining batch dims
        if len(batch_dims1) > len(batch_dims2):
            result_dims = batch_dims1[:len(batch_dims1) - len(batch_dims2)] + result_dims
        elif len(batch_dims2) > len(batch_dims1):
            result_dims = batch_dims2[:len(batch_dims2) - len(batch_dims1)] + result_dims
    elif batch_dims1:
        result_dims.extend(batch_dims1)
    elif batch_dims2:
        result_dims.extend(batch_dims2)
    
    result_dims.extend([dims1[-2], dims2[-1]])
    
    dtype = input1.dtype if input1.dtype.value >= input2.dtype.value else input2.dtype
    
    return TensorShape(dims=result_dims, dtype=dtype)


def infer_unary_op_shape(input: TensorShape) -> TensorShape:
    """Infer output shape for unary operations (same shape as input)."""
    return TensorShape(dims=input.dims.copy(), dtype=input.dtype)


def infer_reduce_shape(input: TensorShape, dims: List[int], keepdims: bool = False) -> TensorShape:
    """Infer output shape for reduction operations."""
    if not dims:
        return TensorShape(dims=input.dims.copy(), dtype=input.dtype)
    
    result_dims = []
    for i, dim in enumerate(input.dims):
        if i in dims or (i - len(input.dims)) in dims:
            if keepdims:
                result_dims.append(1)
        else:
            result_dims.append(dim)
    
    return TensorShape(dims=result_dims, dtype=input.dtype)


def infer_shape(model: ModelIR) -> ModelIR:
    """
    Perform shape inference on the entire model.
    
    Args:
        model: The ModelIR to perform inference on
        
    Returns:
        ModelIR with inferred shapes
        
    Raises:
        ShapeInferenceError: If shape inference fails
    """
    # Build symbol table
    symbol_table: Dict[str, TensorShape] = {}
    
    # Initialize with model inputs
    for name, shape in model.inputs.items():
        symbol_table[name] = shape
    
    # Initialize with constants
    for name, value in model.constants.items():
        # Infer shape from constant value
        if isinstance(value, (int, float)):
            symbol_table[name] = TensorShape(dims=[], dtype=DataType.FLOAT32)
        elif isinstance(value, (list, tuple)):
            symbol_table[name] = TensorShape(dims=[len(value)], dtype=DataType.FLOAT32)
    
    # Process nodes in order
    for node in model.nodes:
        input_shapes = []
        for input_id in node.inputs:
            if input_id not in symbol_table:
                raise ShapeInferenceError(f"Unknown input: {input_id}")
            input_shapes.append(symbol_table[input_id])
        
        # Infer output shape based on operation type
        output_shape = _infer_node_shape(node.op_type, input_shapes, node.attributes)
        
        # Store output shape
        for output_id in node.outputs:
            symbol_table[output_id] = output_shape
            node.shape = output_shape
    
    # Set model outputs
    for name, shape_id in model.outputs.items():
        if shape_id in symbol_table:
            model.outputs[name] = symbol_table[shape_id]
    
    return model


def _infer_node_shape(op_type: str, input_shapes: List[TensorShape], 
                      attributes: Dict[str, Any]) -> TensorShape:
    """Infer shape for a specific operation."""
    
    # Unary operations
    unary_ops = ["neg", "abs", "exp", "log", "sqrt", "sin", "cos", "tan", 
                 "tanh", "sigmoid", "relu", "softmax", "dropout"]
    if op_type.lower() in unary_ops:
        if not input_shapes:
            raise ShapeInferenceError(f"{op_type} requires at least one input")
        return infer_unary_op_shape(input_shapes[0])
    
    # Binary operations
    binary_ops = ["add", "sub", "mul", "div", "pow", "mod", "min", "max"]
    if op_type.lower() in binary_ops:
        if len(input_shapes) < 2:
            raise ShapeInferenceError(f"{op_type} requires two inputs")
        return infer_binary_op_shape(input_shapes[0], input_shapes[1])
    
    # Matrix multiplication
    if op_type.lower() in ["matmul", "mm"]:
        if len(input_shapes) < 2:
            raise ShapeInferenceError("MatMul requires two inputs")
        return infer_matmul_shape(input_shapes[0], input_shapes[1])
    
    # Reduction operations
    reduce_ops = ["sum", "mean", "max", "min", "prod"]
    if op_type.lower() in reduce_ops:
        if not input_shapes:
            raise ShapeInferenceError(f"{op_type} requires at least one input")
        dims = attributes.get("dim", [])
        keepdims = attributes.get("keepdim", False)
        if isinstance(dims, int):
            dims = [dims]
        return infer_reduce_shape(input_shapes[0], dims, keepdims)
    
    # Reshape
    if op_type.lower() == "reshape":
        if not input_shapes:
            raise ShapeInferenceError("Reshape requires an input")
        new_shape = attributes.get("shape", [])
        return TensorShape(dims=new_shape, dtype=input_shapes[0].dtype)
    
    # Transpose
    if op_type.lower() == "transpose":
        if not input_shapes:
            raise ShapeInferenceError("Transpose requires an input")
        perm = attributes.get("perm")
        if perm:
            new_dims = [input_shapes[0].dims[p] for p in perm]
            return TensorShape(dims=new_dims, dtype=input_shapes[0].dtype)
        return infer_unary_op_shape(input_shapes[0])
    
    # Concatenate
    if op_type.lower() == "concat":
        if len(input_shapes) < 2:
            raise ShapeInferenceError("Concat requires at least two inputs")
        dim = attributes.get("dim", 0)
        if dim < 0:
            dim = len(input_shapes[0].dims) + dim
        
        result_dims = input_shapes[0].dims.copy()
        for shape in input_shapes[1:]:
            for i, d in enumerate(shape.dims):
                if i == dim:
                    result_dims[i] += d
                elif i < len(result_dims) and result_dims[i] != d:
                    raise ShapeInferenceError(
                        f"Concat dimension mismatch at dim {i}: {result_dims[i]} vs {d}"
                    )
        
        return TensorShape(dims=result_dims, dtype=input_shapes[0].dtype)
    
    # Default: assume unary
    if input_shapes:
        return infer_unary_op_shape(input_shapes[0])
    
    raise ShapeInferenceError(f"Unknown operation: {op_type}")
