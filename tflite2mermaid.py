import os
import sys
import struct
import argparse

import pandas as pd
import flatbuffers

import tflite.Model
from tflite.BuiltinOperator import BuiltinOperator

# from tflite.BuiltinOperator import BuiltinOperator as OpType
# from tflite.TensorType import TensorType as TType
import numpy as np

# Load tflite flatbuffer to object tree.
def load_model(filename):
    with open(filename, "rb") as f:
        buf = bytearray(f.read())

    model = tflite.Model.GetRootAsModel(buf, 0)
    return model

def tflite_to_mermaid(m):
    print("tflite_to_mermaid")
    tensor2node = {}
    io_nodes = []
    nodes = {}
    edges = {}
    assert m.SubgraphsLength() == 1, "Multi-subgraph models are currently not supported"
    graph = m.Subgraphs(0)
    print("graph", graph)
    for in_idx in range(graph.InputsLength()):
        print("in_idx", in_idx)
        input_tensor_idx = graph.Inputs(in_idx)
        print("input_tensor_idx", input_tensor_idx)
        input_tensor = graph.Tensors(input_tensor_idx)
        input_name = input_tensor.Name().decode()
        print("input_name", input_name)
        node = f"in{in_idx}"
        node_properties = {"label": input_name}
        nodes[node] = node_properties
        tensor2node[input_tensor_idx] = node
        io_nodes.append(node)
    for op_idx in range(graph.OperatorsLength()):
        print("op_idx", op_idx)
        op = graph.Operators(op_idx)
        print("op", op)
        op_code = m.OperatorCodes(op.OpcodeIndex())
        print("op_code", op_code)
        op_code_id = op_code.BuiltinCode()
        print("op_code_id", op_code_id)
        name = tflite.opcode2name(op_code_id)
        print("name", name)
        node = f"op{op_idx}"
        node_properties = {"label": name, "idx": op_idx}
        nodes[node] = node_properties
        for input_idx in range(op.InputsLength()):
            print("input_idx", input_idx)
            tensor_idx = op.Inputs(input_idx)
            print("tensor_idx", tensor_idx)
            tensor = graph.Tensors(tensor_idx)
            print("tensor", tensor)
            tensor_type = tensor.Type()
            print("tensor_type", tensor_type)
            tensor_shape = tensor.ShapeAsNumpy()
            print("tensor_shape", tensor_shape)
            buf = m.Buffers(tensor.Buffer())
            print("buf", buf)
            is_const = buf.DataLength() > 0
            print("is_const", is_const)
            if is_const:
                continue
            edge_properties = {"idx": tensor_idx}
            src_node = tensor2node[tensor_idx]
            edges[(src_node, node)] = edge_properties
        for output_idx in range(op.OutputsLength()):
            print("output_idx", output_idx)
            tensor_idx = op.Outputs(output_idx)
            print("tensor_idx", tensor_idx)
            tensor = graph.Tensors(tensor_idx)
            print("tensor", tensor)
            tensor_type = tensor.Type()
            print("tensor_type", tensor_type)
            tensor_shape = tensor.ShapeAsNumpy()
            print("tensor_shape", tensor_shape)
            buf = m.Buffers(tensor.Buffer())
            print("buf", buf)
            is_const = buf.DataLength() > 0
            print("is_const", is_const)
            tensor2node[tensor_idx] = node
    for out_idx in range(graph.OutputsLength()):
        print("out_idx", out_idx)
        output_tensor_idx = graph.Outputs(out_idx)
        print("output_tensor_idx", output_tensor_idx)
        output_tensor = graph.Tensors(output_tensor_idx)
        output_name = output_tensor.Name().decode()
        print("output_name", output_name)
        node = f"out{out_idx}"
        node_properties = {"label": output_name}
        edge_properties = {"idx": output_tensor_idx}
        nodes[node] = node_properties
        src_node = tensor2node[output_tensor_idx]
        edges[(src_node, node)] = edge_properties
        # tensor2node[input_tensor_idx] = f"in{in_idx}"
        io_nodes.append(node)
    print("======================")
    print("tensor2node", tensor2node)
    print("nodes", nodes)
    print("edges", edges)
    print("io_nodes", io_nodes)
    print("======================")
    out = """%%{init: {\"flowchart\": {\"htmlLabels\": false}} }%%
flowchart TB\n"""
    for node, node_properties in nodes.items():
        label = node_properties.get("label", "?")
        idx = node_properties.get("idx", "?")
        if node in io_nodes:
            out += f"  {node}([{label}])\n"
        else:
            detail = True
            if detail:
                out += f"""  {node}[\"`**{label}**
_idx: {idx}_`\"]\n"""
            else:
                out += f"  {node}[{label}]\n"
    for edge, edge_properties in edges.items():
        src, dst = edge
        # src_properties = nodes[src]
        # dst_properties = nodes[dst]
        label = edge_properties.get("idx", "?")
        # src_label = src_properties.get("label", "?")
        # dst_label = dst_properties.get("label", "?")
        # out += f"  {src}[{src_label}] --> |{label}| {dst}[{dst_label}]\n"
        out += f"  {src} --> |{label}| {dst}\n"
    print(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()
    # print(args)

    m = load_model(args.model)

    estimated_rom = tflite_to_mermaid(m)


if __name__ == "__main__":
    main()
