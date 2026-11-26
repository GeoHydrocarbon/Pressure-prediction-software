"""Read spatial tables with physical source rows and explicit import settings."""
import csv
from itertools import islice
from pathlib import Path
import shlex
import pandas as pd

DEFAULT_IMPORT = dict(header_row=1, data_row=2, delimiter='auto', encoding='auto')


def settings(options=None):
    options = dict(DEFAULT_IMPORT, **(options or {}))
    for key in ('header_row', 'data_row'):
        value = options[key]
        if isinstance(value,bool) or not isinstance(value,int) or value < (0 if key=='header_row' else 1):
            raise ValueError('表头行必须为非负整数，数据起始行必须为正整数')
    if options['header_row'] and options['data_row'] <= options['header_row']:
        raise ValueError('数据起始行必须在表头行之后')
    if options['delimiter'] not in ('auto','whitespace','\t',',',';','|'):
        raise ValueError('不支持该分隔符')
    if options['encoding'] not in ('auto','utf-8-sig','gb18030'):
        raise ValueError('不支持该文本编码')
    return options


def text_encoding(path, requested='auto'):
    if requested != 'auto': return requested
    for encoding in ('utf-8-sig','gb18030'):
        try:
            with Path(path).open(encoding=encoding) as stream:
                for _ in islice(stream,80): pass
            return encoding
        except UnicodeDecodeError: pass
    raise ValueError('文本不是 UTF-8 或 GB18030 编码，请另存为 UTF-8')


def preview_source(path, encoding='auto', limit=60):
    path = Path(path)
    if path.suffix.lower()=='.xlsx':
        from openpyxl import load_workbook
        book=load_workbook(path,read_only=True,data_only=True)
        try:
            return [(i,' | '.join('' if v is None else str(v) for v in row))
                    for i,row in enumerate(book.active.iter_rows(max_row=limit,values_only=True),1)]
        finally: book.close()
    if path.suffix.lower() not in ('.csv','.txt'): raise ValueError('空间数据支持 .xlsx、.csv、.txt')
    with path.open(encoding=text_encoding(path,encoding),newline='') as stream:
        return [(i,line.rstrip('\r\n')) for i,line in enumerate(islice(stream,limit),1)]


def records(path, options):
    """Yield starting physical row and fields, excluding preamble before header."""
    first=options['header_row'] or options['data_row']
    if path.suffix.lower()=='.xlsx':
        from openpyxl import load_workbook
        book=load_workbook(path,read_only=True,data_only=True)
        try:
            for i,row in enumerate(book.active.iter_rows(min_row=first,values_only=True),first):
                yield i,list(row)
        finally: book.close()
        return
    if path.suffix.lower() not in ('.csv','.txt'): raise ValueError('空间数据支持 .xlsx、.csv、.txt')
    with path.open(encoding=text_encoding(path,options['encoding']),newline='') as stream:
        for _ in range(first-1):
            if not stream.readline(): return
        delimiter=options['delimiter']
        if delimiter=='auto':
            position=stream.tell();sample=''.join(islice(stream,12));stream.seek(position)
            # A header is sufficient when explanatory rows follow it.
            header=sample.splitlines()[0] if sample else ''
            try: delimiter=csv.Sniffer().sniff(header,delimiters=',\t;|').delimiter
            except csv.Error: delimiter='whitespace'
        if delimiter=='whitespace':
            for i,line in enumerate(stream,first):
                yield i,shlex.split(line,comments=False,posix=True) if line.strip() else []
        else:
            reader=csv.reader(stream,delimiter=delimiter);previous=0
            for row in reader:
                start=first+previous;previous=reader.line_num;yield start,row


def read_spatial_table(path, options=None, nrows=None):
    path=Path(path);options=settings(options);header=None;data=[];source=[]
    iterator=records(path,options)
    try:
        for row,values in iterator:
            if row==options['header_row']:
                header=[str(v).strip() if v is not None and str(v).strip() else f'Column_{i}' for i,v in enumerate(values,1)]
                if len(header)!=len(set(header)): raise ValueError('表头存在重复列名，请选择正确的表头行或修改文件')
                continue
            if row<options['data_row'] or not values or all(v is None or str(v).strip()=='' for v in values): continue
            if header is None:
                if options['header_row']: raise ValueError('未找到指定的表头行')
                header=[f'Column_{i}' for i in range(1,len(values)+1)]
            if len(values)>len(header):
                if any(v is not None and str(v).strip() for v in values[len(header):]):
                    raise ValueError(f'第 {row} 行列数多于表头，请检查分隔符和行号')
                values=values[:len(header)]
            data.append(list(values)+[None]*(len(header)-len(values)));source.append(row)
            if nrows is not None and len(data)>=nrows: break
    finally: iterator.close()
    if header is None: raise ValueError('没有读到表头或数据，请调整行号和分隔符')
    frame=pd.DataFrame(data,columns=header)
    frame.attrs['source_rows']=source
    return frame
