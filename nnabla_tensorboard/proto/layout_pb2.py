# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nnabla_tensorboard/proto/layout.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='nnabla_tensorboard/proto/layout.proto',
  package='tensorboard',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n%nnabla_tensorboard/proto/layout.proto\x12\x0btensorboard\"\x8d\x01\n\x05\x43hart\x12\r\n\x05title\x18\x01 \x01(\t\x12\x37\n\tmultiline\x18\x02 \x01(\x0b\x32\".tensorboard.MultilineChartContentH\x00\x12\x31\n\x06margin\x18\x03 \x01(\x0b\x32\x1f.tensorboard.MarginChartContentH\x00\x42\t\n\x07\x63ontent\"$\n\x15MultilineChartContent\x12\x0b\n\x03tag\x18\x01 \x03(\t\"\x83\x01\n\x12MarginChartContent\x12\x36\n\x06series\x18\x01 \x03(\x0b\x32&.tensorboard.MarginChartContent.Series\x1a\x35\n\x06Series\x12\r\n\x05value\x18\x01 \x01(\t\x12\r\n\x05lower\x18\x02 \x01(\t\x12\r\n\x05upper\x18\x03 \x01(\t\"L\n\x08\x43\x61tegory\x12\r\n\x05title\x18\x01 \x01(\t\x12!\n\x05\x63hart\x18\x02 \x03(\x0b\x32\x12.tensorboard.Chart\x12\x0e\n\x06\x63losed\x18\x03 \x01(\x08\"B\n\x06Layout\x12\x0f\n\x07version\x18\x01 \x01(\x05\x12\'\n\x08\x63\x61tegory\x18\x02 \x03(\x0b\x32\x15.tensorboard.Categoryb\x06proto3')
)




_CHART = _descriptor.Descriptor(
  name='Chart',
  full_name='tensorboard.Chart',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='title', full_name='tensorboard.Chart.title', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='multiline', full_name='tensorboard.Chart.multiline', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='margin', full_name='tensorboard.Chart.margin', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
    _descriptor.OneofDescriptor(
      name='content', full_name='tensorboard.Chart.content',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=55,
  serialized_end=196,
)


_MULTILINECHARTCONTENT = _descriptor.Descriptor(
  name='MultilineChartContent',
  full_name='tensorboard.MultilineChartContent',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='tag', full_name='tensorboard.MultilineChartContent.tag', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=198,
  serialized_end=234,
)


_MARGINCHARTCONTENT_SERIES = _descriptor.Descriptor(
  name='Series',
  full_name='tensorboard.MarginChartContent.Series',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='tensorboard.MarginChartContent.Series.value', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='lower', full_name='tensorboard.MarginChartContent.Series.lower', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='upper', full_name='tensorboard.MarginChartContent.Series.upper', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
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
  serialized_start=315,
  serialized_end=368,
)

_MARGINCHARTCONTENT = _descriptor.Descriptor(
  name='MarginChartContent',
  full_name='tensorboard.MarginChartContent',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='series', full_name='tensorboard.MarginChartContent.series', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_MARGINCHARTCONTENT_SERIES, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=237,
  serialized_end=368,
)


_CATEGORY = _descriptor.Descriptor(
  name='Category',
  full_name='tensorboard.Category',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='title', full_name='tensorboard.Category.title', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='chart', full_name='tensorboard.Category.chart', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='closed', full_name='tensorboard.Category.closed', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
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
  serialized_start=370,
  serialized_end=446,
)


_LAYOUT = _descriptor.Descriptor(
  name='Layout',
  full_name='tensorboard.Layout',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='version', full_name='tensorboard.Layout.version', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='category', full_name='tensorboard.Layout.category', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=448,
  serialized_end=514,
)

_CHART.fields_by_name['multiline'].message_type = _MULTILINECHARTCONTENT
_CHART.fields_by_name['margin'].message_type = _MARGINCHARTCONTENT
_CHART.oneofs_by_name['content'].fields.append(
  _CHART.fields_by_name['multiline'])
_CHART.fields_by_name['multiline'].containing_oneof = _CHART.oneofs_by_name['content']
_CHART.oneofs_by_name['content'].fields.append(
  _CHART.fields_by_name['margin'])
_CHART.fields_by_name['margin'].containing_oneof = _CHART.oneofs_by_name['content']
_MARGINCHARTCONTENT_SERIES.containing_type = _MARGINCHARTCONTENT
_MARGINCHARTCONTENT.fields_by_name['series'].message_type = _MARGINCHARTCONTENT_SERIES
_CATEGORY.fields_by_name['chart'].message_type = _CHART
_LAYOUT.fields_by_name['category'].message_type = _CATEGORY
DESCRIPTOR.message_types_by_name['Chart'] = _CHART
DESCRIPTOR.message_types_by_name['MultilineChartContent'] = _MULTILINECHARTCONTENT
DESCRIPTOR.message_types_by_name['MarginChartContent'] = _MARGINCHARTCONTENT
DESCRIPTOR.message_types_by_name['Category'] = _CATEGORY
DESCRIPTOR.message_types_by_name['Layout'] = _LAYOUT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Chart = _reflection.GeneratedProtocolMessageType('Chart', (_message.Message,), dict(
  DESCRIPTOR = _CHART,
  __module__ = 'nnabla_tensorboard.proto.layout_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.Chart)
  ))
_sym_db.RegisterMessage(Chart)

MultilineChartContent = _reflection.GeneratedProtocolMessageType('MultilineChartContent', (_message.Message,), dict(
  DESCRIPTOR = _MULTILINECHARTCONTENT,
  __module__ = 'nnabla_tensorboard.proto.layout_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.MultilineChartContent)
  ))
_sym_db.RegisterMessage(MultilineChartContent)

MarginChartContent = _reflection.GeneratedProtocolMessageType('MarginChartContent', (_message.Message,), dict(

  Series = _reflection.GeneratedProtocolMessageType('Series', (_message.Message,), dict(
    DESCRIPTOR = _MARGINCHARTCONTENT_SERIES,
    __module__ = 'nnabla_tensorboard.proto.layout_pb2'
    # @@protoc_insertion_point(class_scope:tensorboard.MarginChartContent.Series)
    ))
  ,
  DESCRIPTOR = _MARGINCHARTCONTENT,
  __module__ = 'nnabla_tensorboard.proto.layout_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.MarginChartContent)
  ))
_sym_db.RegisterMessage(MarginChartContent)
_sym_db.RegisterMessage(MarginChartContent.Series)

Category = _reflection.GeneratedProtocolMessageType('Category', (_message.Message,), dict(
  DESCRIPTOR = _CATEGORY,
  __module__ = 'nnabla_tensorboard.proto.layout_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.Category)
  ))
_sym_db.RegisterMessage(Category)

Layout = _reflection.GeneratedProtocolMessageType('Layout', (_message.Message,), dict(
  DESCRIPTOR = _LAYOUT,
  __module__ = 'nnabla_tensorboard.proto.layout_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.Layout)
  ))
_sym_db.RegisterMessage(Layout)


# @@protoc_insertion_point(module_scope)
