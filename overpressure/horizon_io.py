"""Bounded text-table reader for TXT/UDF horizon exports, with explicit parsing options."""
from array import array
import codecs
import csv
import io
from pathlib import Path
import re
import shlex
import numpy as np
import pandas as pd
from .data import read_table

MAX_HORIZON_RECORDS=3_000_000
SEPARATORS={'whitespace':r'\s+','comma':',','tab':'\t','semicolon':';','pipe':'|'}


class HorizonTextStream(io.TextIOBase):
    def __init__(self,path,encoding,skip):
        self.file=open(path,'r',encoding=encoding,newline='');self.number=0;self.rows=array('q');self.buffer=''
        for _ in range(skip):
            if not self.file.readline():break
            self.number+=1
    def readable(self):return True
    def readline(self,size=-1):
        while True:
            line=self.file.readline()
            if not line:return ''
            self.number+=1;clean=line.strip()
            if clean and not clean.startswith(('#','!','//')):
                self.rows.append(self.number);return line
    def read(self,size=-1):
        if size<0:
            prefix=self.buffer;self.buffer='';return prefix+''.join(self)
        pieces=[self.buffer];length=len(self.buffer)
        while length<size:
            line=self.readline()
            if not line:break
            pieces.append(line);length+=len(line)
        joined=''.join(pieces);value,self.buffer=joined[:size],joined[size:];return value
    def close(self):
        self.file.close();super().close()


def text_encoding(path):
    with open(path,'rb') as source:sample=source.read(131072)
    if sample.startswith((b'\xff\xfe',b'\xfe\xff')):return 'utf-16'
    if b'\x00' in sample:raise ValueError('该文件不是可识别的文本表格。UDF 当前支持按列排列的文本数据；专有或二进制结构需要样例适配。')
    for encoding in ('utf-8-sig','gb18030'):
        try:codecs.getincrementaldecoder(encoding)().decode(sample,final=False);return encoding
        except UnicodeDecodeError:continue
    raise ValueError('无法识别文本编码，请使用 UTF-8、UTF-16 或 GB18030 文本文件')


def read_horizon_table(path,options=None,*,nrows=None):
    path=Path(path);options=options or {}
    if path.suffix.lower() in ('.csv','.xlsx'):return read_table(path,nrows=nrows)
    if path.suffix.lower() not in ('.txt','.udf'):raise ValueError('层位文件支持 CSV、XLSX、TXT 和文本表格型 UDF')
    delimiter=options.get('delimiter','auto');header=options.get('header','auto');skip=options.get('skip_rows',0)
    if delimiter not in ('auto',*SEPARATORS) or header not in ('auto','present','none'):raise ValueError('文本分隔符或列名设置无效')
    if isinstance(skip,bool) or not isinstance(skip,int) or not 0<=skip<=100000:raise ValueError('跳过开头行数必须为 0～100000 的整数')
    encoding=text_encoding(path)
    with HorizonTextStream(path,encoding,skip) as stream:
        sample=[stream.readline() for _ in range(12)]
    sample=[line for line in sample if line]
    if not sample:raise ValueError('跳过说明行后没有可读取的数据')
    if delimiter=='auto':
        try:separator=csv.Sniffer().sniff(''.join(sample),delimiters=',\t;|').delimiter
        except csv.Error:separator=r'\s+'
    else:separator=SEPARATORS[delimiter]
    fields=shlex.split(sample[0].strip()) if separator==r'\s+' else next(csv.reader([sample[0]],delimiter=separator))
    numeric=0
    for field in fields:
        try:float(field);numeric+=1
        except ValueError:pass
    has_header=header=='present' or header=='auto' and numeric==0
    names=None if has_header else [f'第{i+1}列' for i in range(len(fields))]
    try:
        with HorizonTextStream(path,encoding,skip) as stream:
            frame=pd.read_csv(stream,sep=separator,engine='c',header=0 if has_header else None,names=names,
                              nrows=MAX_HORIZON_RECORDS+1 if nrows is None else nrows,skipinitialspace=True)
            source_rows=np.asarray(stream.rows,dtype=np.int64)[int(has_header):int(has_header)+len(frame)].copy()
    except (pd.errors.ParserError,UnicodeError,pd.errors.EmptyDataError) as exc:
        raise ValueError('文本列结构无法读取，请核对分隔符、是否有列名及跳过开头行数。原始错误：'+str(exc)) from exc
    if len(frame)>MAX_HORIZON_RECORDS:raise ValueError('层位文件超过 300 万行，请拆分后导入')
    if nrows is None and len(stream.rows)!=len(frame)+int(has_header):raise ValueError('层位文本字段不能跨多行；请将层位名称等文本整理在同一行后导入')
    if len(frame.columns)<3:raise ValueError('未识别到至少三列坐标/时间数据，请核对分隔符或跳过说明行')
    frame.attrs.update(source_rows=source_rows,text_header=has_header,text_separator=separator,text_encoding=encoding)
    return frame
