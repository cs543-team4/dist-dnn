# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: inference_service.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor.FileDescriptor(
    name='inference_service.proto',
    package='tensorflow',
    syntax='proto3',
    serialized_options=b'\n\030org.tensorflow.frameworkB\014TensorProtosP\001Z=github.com/tensorflow/tensorflow/tensorflow/go/core/framework\370\001\001',
    serialized_pb=b'\n\x17inference_service.proto\x12\ntensorflow\" \n\x10SerializedTensor\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\t\"\x18\n\x05Reply\x12\x0f\n\x07message\x18\x01 \x01(\t2V\n\x10InferenceService\x12\x42\n\rProcessTensor\x12\x1c.tensorflow.SerializedTensor\x1a\x11.tensorflow.Reply\"\x00\x42l\n\x18org.tensorflow.frameworkB\x0cTensorProtosP\x01Z=github.com/tensorflow/tensorflow/tensorflow/go/core/framework\xf8\x01\x01\x62\x06proto3'
)

_SERIALIZEDTENSOR = _descriptor.Descriptor(
    name='SerializedTensor',
    full_name='tensorflow.SerializedTensor',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='data', full_name='tensorflow.SerializedTensor.data', index=0,
            number=1, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=39,
    serialized_end=71,
)

_REPLY = _descriptor.Descriptor(
    name='Reply',
    full_name='tensorflow.Reply',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='message', full_name='tensorflow.Reply.message', index=0,
            number=1, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=73,
    serialized_end=97,
)

DESCRIPTOR.message_types_by_name['SerializedTensor'] = _SERIALIZEDTENSOR
DESCRIPTOR.message_types_by_name['Reply'] = _REPLY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SerializedTensor = _reflection.GeneratedProtocolMessageType('SerializedTensor', (_message.Message,), {
    'DESCRIPTOR': _SERIALIZEDTENSOR,
    '__module__': 'inference_service_pb2'
    # @@protoc_insertion_point(class_scope:tensorflow.SerializedTensor)
})
_sym_db.RegisterMessage(SerializedTensor)

Reply = _reflection.GeneratedProtocolMessageType('Reply', (_message.Message,), {
    'DESCRIPTOR': _REPLY,
    '__module__': 'inference_service_pb2'
    # @@protoc_insertion_point(class_scope:tensorflow.Reply)
})
_sym_db.RegisterMessage(Reply)

DESCRIPTOR._options = None

_INFERENCESERVICE = _descriptor.ServiceDescriptor(
    name='InferenceService',
    full_name='tensorflow.InferenceService',
    file=DESCRIPTOR,
    index=0,
    serialized_options=None,
    serialized_start=99,
    serialized_end=185,
    methods=[
        _descriptor.MethodDescriptor(
            name='ProcessTensor',
            full_name='tensorflow.InferenceService.ProcessTensor',
            index=0,
            containing_service=None,
            input_type=_SERIALIZEDTENSOR,
            output_type=_REPLY,
            serialized_options=None,
        ),
    ])
_sym_db.RegisterServiceDescriptor(_INFERENCESERVICE)

DESCRIPTOR.services_by_name['InferenceService'] = _INFERENCESERVICE

# @@protoc_insertion_point(module_scope)
