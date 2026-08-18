import io
import re

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="TÙNG Data Studio", page_icon="📊", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:1.2rem;max-width:1500px}
.hero{border:1px solid rgba(130,140,155,.22);border-radius:16px;padding:18px 22px;margin-bottom:14px;background:rgba(255,255,255,.015)}
.brand{font-size:2.35rem;font-weight:800;letter-spacing:-.04em}
.sub{color:#8f96a3;margin-top:5px}
div[data-testid="stMetric"]{border:1px solid rgba(130,140,155,.22);border-radius:12px;padding:12px 14px;background:rgba(255,255,255,.018)}
[data-testid="stSidebar"]{border-right:1px solid rgba(125,135,150,.15)}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="brand">TÙNG</div>
  <div style="font-size:1.08rem;font-weight:650">Data Studio</div>
  <div class="sub">Kiểm tra chất lượng · Làm sạch · Phân tích · Truy xuất · Xuất dữ liệu</div>
</div>
""", unsafe_allow_html=True)

NULL_TOKENS={"","null","none","n/a","na","nan","-","--","missing","unknown"}
CURRENCY_CODES=r"(?:USD|EUR|GBP|JPY|VND|AUD|CAD)"
CURRENCY_SYMBOLS=r"[$€£¥]"

@st.cache_data(show_spinner=False)
def doc_file(data:bytes,name:str):
    n=name.lower()
    if n.endswith('.csv'):
        err=None
        for enc in ('utf-8-sig','utf-8','latin1'):
            try:return pd.read_csv(io.BytesIO(data),encoding=enc,sep=None,engine='python')
            except Exception as e:err=e
        raise ValueError(err)
    if n.endswith(('.xlsx','.xls')):return pd.read_excel(io.BytesIO(data))
    if n.endswith('.parquet'):return pd.read_parquet(io.BytesIO(data))
    raise ValueError('Định dạng chưa được hỗ trợ')

def snake(x):
    x=re.sub(r'[\s\-/]+','_',str(x).strip())
    x=re.sub(r'[^0-9a-zA-Z_]+','',x)
    return re.sub(r'_+','_',x).strip('_').lower() or 'column'

def unique_cols(cols):
    seen={};out=[]
    for c in cols:
        k=seen.get(c,0);out.append(c if k==0 else f'{c}_{k}');seen[c]=k+1
    return out

def parse_number(v):
    if pd.isna(v):return np.nan
    s=str(v).strip()
    s=re.sub(CURRENCY_CODES,'',s,flags=re.I)
    s=re.sub(CURRENCY_SYMBOLS,'',s)
    s=re.sub(r'\s+','',s)
    if not s:return np.nan
    if ',' in s and '.' in s:
        s=s.replace('.','').replace(',','.') if s.rfind(',')>s.rfind('.') else s.replace(',','')
    elif ',' in s:
        last=s.split(',')[-1]
        s=s.replace('.','').replace(',','.') if len(last) in (1,2) else s.replace(',','')
    try:return float(s)
    except:return np.nan

def parse_date(s):
    try:
        a=pd.to_datetime(s,errors='coerce',format='mixed',dayfirst=True)
        b=pd.to_datetime(s,errors='coerce',format='mixed',dayfirst=False)
    except TypeError:
        a=pd.to_datetime(s,errors='coerce',dayfirst=True)
        b=pd.to_datetime(s,errors='coerce',dayfirst=False)
    return a if a.notna().sum()>=b.notna().sum() else b

def clean_data(df):
    out=df.copy();log=[]
    old=list(out.columns);out.columns=unique_cols([snake(c) for c in old])
    if old!=list(out.columns):log.append(['Chuẩn hóa cột','Tất cả','Đổi tên cột sang snake_case'])
    for c in out.select_dtypes(include='object').columns:
        before=out[c].copy()
        out[c]=out[c].map(lambda x:re.sub(r'\s+',' ',x).strip() if isinstance(x,str) else x)
        out[c]=out[c].map(lambda x:np.nan if isinstance(x,str) and x.lower() in NULL_TOKENS else x)
        changed=int((before.fillna('__NA__').astype(str)!=out[c].fillna('__NA__').astype(str)).sum())
        if changed:log.append(['Chuẩn hóa text',c,f'{changed} giá trị được chuẩn hóa'])
    dup=int(out.duplicated().sum())
    if dup:
        out=out.drop_duplicates().reset_index(drop=True);log.append(['Xóa trùng','Dòng',f'Đã xóa {dup} dòng trùng hoàn toàn'])
    for c in list(out.select_dtypes(include='object').columns):
        s=out[c];sample=s.dropna().astype(str).head(800)
        if len(sample)<3:continue
        name=c.lower();id_like=any(k in name for k in ('id','code','zip','postal','phone'))
        explicit=sample.str.contains(r'[$€£¥]|\b(?:USD|EUR|GBP|JPY|VND|AUD|CAD)\b',case=False,regex=True).mean()
        numeric_like=sample.map(lambda x:pd.notna(parse_number(x))).mean()
        date_like=parse_date(sample).notna().mean()
        date_name=any(k in name for k in ('date','time','occur','declar','created','updated','birth'))
        if not id_like and explicit>=.2 and numeric_like>=.9:
            parsed=s.map(parse_number);fail=int((s.notna()&parsed.isna()).sum());out[c]=parsed
            log.append(['Chuyển kiểu dữ liệu',c,f'Tiền tệ/text → số; lỗi parse: {fail}'])
        elif not id_like and numeric_like>=.95:
            parsed=s.map(parse_number);fail=int((s.notna()&parsed.isna()).sum());out[c]=parsed
            log.append(['Chuyển kiểu dữ liệu',c,f'Text → số; lỗi parse: {fail}'])
        elif (date_name and date_like>=.6) or date_like>=.93:
            parsed=parse_date(s);fail=int((s.notna()&parsed.isna()).sum());out[c]=parsed
            log.append(['Chuyển kiểu dữ liệu',c,f'Text → ngày; lỗi parse: {fail}'])
    occur=next((c for c in out.columns if 'occur' in c and pd.api.types.is_datetime64_any_dtype(out[c])),None)
    declar=next((c for c in out.columns if 'declar' in c and pd.api.types.is_datetime64_any_dtype(out[c])),None)
    if occur and declar:
        out['claim_delay_days']=(out[declar]-out[occur]).dt.days
        log.append(['Tạo cột phân tích','claim_delay_days',f'{declar} - {occur}'])
    damage=next((c for c in out.columns if 'damage' in c and pd.api.types.is_numeric_dtype(out[c])),None)
    indemn=next((c for c in out.columns if 'indemn' in c and pd.api.types.is_numeric_dtype(out[c])),None)
    if damage and indemn:
        out['indemnification_rate']=out[indemn]/out[damage].replace(0,np.nan)
        log.append(['Tạo cột phân tích','indemnification_rate',f'{indemn} / {damage}'])
    return out,pd.DataFrame(log,columns=['Bước','Cột','Chi tiết'])

def profile(df):
    return pd.DataFrame([{'Cột':c,'Kiểu dữ liệu':str(df[c].dtype),'Missing':int(df[c].isna().sum()),'Missing %':round(df[c].isna().mean()*100,2),'Unique':int(df[c].nunique(dropna=True)),'Mẫu':' | '.join(df[c].dropna().astype(str).head(3).tolist())} for c in df.columns])

def validation(df):
    rows=[]
    dates=[c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    for a in dates:
        for b in dates:
            if a!=b and any(k in a for k in ('occur','incident','order','start')) and any(k in b for k in ('declar','report','ship','end')):
                v=int((df[a].notna()&df[b].notna()&(df[b]<df[a])).sum())
                rows.append({'Quy tắc':f'{b} >= {a}','Vi phạm':v,'Trạng thái':'Đạt' if v==0 else 'Cần xem xét'})
    for c in df.select_dtypes(include=np.number).columns:
        if any(k in c for k in ('amount','sales','revenue','price','cost','premium','damage','indemn')):
            v=int((df[c]<0).sum());rows.append({'Quy tắc':f'{c} >= 0','Vi phạm':v,'Trạng thái':'Đạt' if v==0 else 'Cần xem xét'})
    return pd.DataFrame(rows or [{'Quy tắc':'Chưa phát hiện quy tắc nghiệp vụ tự động','Vi phạm':0,'Trạng thái':'Thông tin'}])

def compact(v):
    if pd.isna(v):return '—'
    v=float(v)
    if abs(v)>=1e9:return f'{v/1e9:.2f}B'
    if abs(v)>=1e6:return f'{v/1e6:.2f}M'
    if abs(v)>=1e3:return f'{v/1e3:.2f}K'
    return f'{v:,.2f}'

def excel_bytes(df,prof,log,val):
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine='openpyxl') as w:
        df.to_excel(w,index=False,sheet_name='Du_lieu_da_lam_sach')
        prof.to_excel(w,index=False,sheet_name='Ho_so_du_lieu')
        val.to_excel(w,index=False,sheet_name='Kiem_tra_nghiep_vu')
        log.to_excel(w,index=False,sheet_name='Nhat_ky_lam_sach')
    return buf.getvalue()

with st.sidebar:
    st.markdown('### TÙNG Data Studio')
    st.caption('Workspace phân tích dữ liệu')
    st.divider()
    uploaded=st.file_uploader('Tải dữ liệu lên',type=['csv','xlsx','xls','parquet'])

if not uploaded:
    st.info('Tải lên một file CSV, Excel hoặc Parquet để bắt đầu.')
    st.stop()

try:raw=doc_file(uploaded.getvalue(),uploaded.name)
except Exception as e:
    st.error(f'Không thể đọc file: {e}');st.stop()

cleaned,log=clean_data(raw)
prof=profile(cleaned);val=validation(cleaned)
nums=cleaned.select_dtypes(include=np.number).columns.tolist()
dims=[c for c in cleaned.select_dtypes(include=['object','bool']).columns if 2<=cleaned[c].nunique(dropna=True)<=100]
dates=[c for c in cleaned.columns if pd.api.types.is_datetime64_any_dtype(cleaned[c])]

with st.sidebar:
    st.divider();st.subheader('Phân tích')
    metric=st.selectbox('Chỉ số (Metric)',['(Không chọn)']+nums);metric=None if metric=='(Không chọn)' else metric
    dim=st.selectbox('Chiều phân tích (Dimension)',['(Không chọn)']+dims);dim=None if dim=='(Không chọn)' else dim
    dcol=st.selectbox('Cột ngày',['(Không chọn)']+dates);dcol=None if dcol=='(Không chọn)' else dcol
    top_n=st.slider('Top N',5,25,10)

st.markdown('**Quy trình:** 1. Tải dữ liệu → 2. Kiểm tra → 3. Làm sạch → 4. Phân tích → 5. Truy xuất → 6. Xuất dữ liệu')

c1,c2,c3,c4,c5,c6=st.columns(6)
c1.metric('Số dòng',f'{len(cleaned):,}');c2.metric('Số cột',cleaned.shape[1]);c3.metric('Ô bị thiếu',int(cleaned.isna().sum().sum()));c4.metric('Dòng trùng',int(cleaned.duplicated().sum()));c5.metric('Cột ngày',len(dates));c6.metric('Kích thước',f'{len(uploaded.getvalue())/(1024*1024):.1f} MB')

tabs=st.tabs(['Tổng quan','Chất lượng dữ liệu','Làm sạch dữ liệu','Phân tích','Truy xuất dữ liệu','Xuất dữ liệu'])

with tabs[0]:
    st.subheader('Tổng quan dữ liệu');st.dataframe(cleaned.head(200),use_container_width=True,height=430)
    if nums:
        st.subheader('Thống kê dữ liệu số');st.dataframe(cleaned[nums].describe().T,use_container_width=True)

with tabs[1]:
    st.subheader('Hồ sơ từng cột');st.dataframe(prof,use_container_width=True,height=420)
    st.subheader('Kiểm tra quy tắc nghiệp vụ');st.dataframe(val,use_container_width=True)

with tabs[2]:
    st.subheader('Làm sạch dữ liệu')
    st.write('Tool tự động áp dụng các bước an toàn khi độ tin cậy cao: chuẩn hóa tên cột, chuẩn hóa text/null, xóa dòng trùng hoàn toàn, nhận diện ngày hỗn hợp, chuyển tiền tệ/số dạng text sang kiểu số và tạo trường phân tích phù hợp.')
    st.markdown('#### Nhật ký làm sạch')
    if len(log):st.dataframe(log,use_container_width=True,height=360)
    else:st.success('Không cần thay đổi làm sạch bổ sung.')

with tabs[3]:
    st.subheader('Phân tích khám phá dữ liệu')
    if metric:
        s=cleaned[metric]
        a,b,c,d=st.columns(4);a.metric(f'Tổng {metric}',compact(s.sum()));b.metric(f'Trung bình {metric}',compact(s.mean()));c.metric(f'Median {metric}',compact(s.median()));d.metric(f'Lớn nhất {metric}',compact(s.max()))
        st.plotly_chart(px.histogram(cleaned,x=metric,nbins=40,title=f'Phân phối của {metric}'),use_container_width=True)
    if metric and dim:
        g=cleaned.groupby(dim,dropna=False)[metric].sum().sort_values(ascending=False).head(top_n).reset_index();fig=px.bar(g,x=metric,y=dim,orientation='h',title=f'Top {top_n} {dim} theo {metric}');fig.update_layout(yaxis={'categoryorder':'total ascending'});st.plotly_chart(fig,use_container_width=True)
    if metric and dcol:
        tmp=cleaned[[dcol,metric]].dropna().set_index(dcol).resample('ME')[metric].sum().reset_index();st.plotly_chart(px.line(tmp,x=dcol,y=metric,markers=True,title=f'{metric} theo thời gian'),use_container_width=True)
    if len(nums)>=2:
        corr=cleaned[nums].corr(numeric_only=True);st.plotly_chart(px.imshow(corr,text_auto='.2f',aspect='auto',title='Ma trận tương quan'),use_container_width=True)

with tabs[4]:
    st.subheader('Truy xuất dữ liệu bằng SQL');st.caption('Tên bảng: `data`')
    sql=st.text_area('Câu lệnh SQL',value='SELECT * FROM data LIMIT 50',height=150)
    if st.button('Chạy SQL'):
        try:
            con=duckdb.connect(database=':memory:');con.register('data',cleaned);result=con.execute(sql).df();con.close();st.success(f'Trả về {len(result):,} dòng.');st.dataframe(result,use_container_width=True,height=360);st.download_button('Tải kết quả truy vấn',result.to_csv(index=False).encode('utf-8-sig'),'query_result.csv','text/csv')
        except Exception as e:st.error(f'Lỗi SQL: {e}')
    with st.expander('Ví dụ SQL'):
        st.code('SELECT status, COUNT(*) AS so_luong\nFROM data\nGROUP BY status\nORDER BY so_luong DESC;',language='sql')

with tabs[5]:
    st.subheader('Xuất dữ liệu')
    st.download_button('Tải dữ liệu đã làm sạch (CSV)',cleaned.to_csv(index=False).encode('utf-8-sig'),'cleaned_data.csv','text/csv',use_container_width=True)
    st.download_button('Tải workbook phân tích (Excel)',excel_bytes(cleaned,prof,log,val),'tung_data_studio_output.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',use_container_width=True)
    if len(log):st.download_button('Tải nhật ký làm sạch',log.to_csv(index=False).encode('utf-8-sig'),'cleaning_log.csv','text/csv',use_container_width=True)

st.markdown('---');st.caption('TÙNG Data Studio · Workspace phân tích và làm sạch dữ liệu')
