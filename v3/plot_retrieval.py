"""Plot saved query/results without changing or rerunning retrieval."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from model import znorm
from index import SearchIndex

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--query',default='../results/query.json')
    ap.add_argument('--result',default='results/cascade_certified_http/original_query.json')
    ap.add_argument('--index',default='results/validation_index')
    ap.add_argument('--output',default='results/retrieval_visualization')
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    fonts={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ('Microsoft YaHei','SimHei','Noto Sans CJK SC') if f in fonts),'DejaVu Sans')
    plt.rcParams.update({'font.family':font,'axes.unicode_minus':False,'font.size':11,
        'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
    query=json.loads(Path(args.query).read_text(encoding='utf-8'))
    saved=json.loads(Path(args.result).read_text(encoding='utf-8'));result=saved.get('result',saved)
    q=np.array(query['values'] if isinstance(query,dict) else query,dtype=float)
    hits=result['hits'][:10]
    if not hits:raise ValueError('No retrieved hits to visualize')
    idx=SearchIndex(args.index)
    for hit in hits:
        if 'values' not in hit:hit['values']=idx.values(hit['sid'])[hit['local_start']:hit['local_start']+len(q)].tolist()
        if len(hit['values'])!=len(q):raise ValueError('Query/result lengths differ')
    h=hits[0];v=np.array(h['values']);t=np.arange(len(q));blue='#2563eb';orange='#ea580c'
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight')
        fig.savefig(out/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(3,1,figsize=(12,8),sharex=True,gridspec_kw={'height_ratios':[1,1,.65]})
    ax[0].plot(t,q,color=blue,lw=2);ax[0].set_title('需要检索的原始片段：244 点',loc='left');ax[0].set_ylabel('原始数值')
    ax[1].plot(t,q,color=blue,lw=3,alpha=.6,label='查询片段')
    ax[1].plot(t,v,color=orange,lw=1.8,ls='--',label='Top-1 检索结果')
    ax[1].set_title(f"Top-1 叠加比较 · 起点 {h['start']} · 标准化欧氏距离 {h['distance']:.6f}",loc='left')
    ax[1].set_ylabel('原始数值');ax[1].legend(loc='best',ncol=2)
    ax[2].plot(t,v-q,color='#059669',lw=1.6);ax[2].axhline(0,color='#64748b',lw=.8)
    err=float(np.max(np.abs(v-q)))
    if err==0:ax[2].set_ylim(-.05,.05)
    ax[2].set_title(f'逐点差值（检索结果 − 查询） · 最大绝对误差 {err:.6g}',loc='left')
    ax[2].set(xlabel='片段内相对采样点（0 起算）',ylabel='差值')
    fig.suptitle('查询与最佳匹配对比',fontsize=18,y=.99)
    fig.text(.08,.008,f"文件：{Path(h['source']).name}   列：{h['column']}   设备：{h.get('device','—')}",fontsize=10)
    fig.tight_layout(rect=(0,.035,1,.965));save(fig,'query_vs_best')

    fig,axes=plt.subplots(5,2,figsize=(15,14),sharex=True,sharey=True)
    zq=znorm(q)
    for rank,(a,hit) in enumerate(zip(axes.flat,hits),1):
        a.plot(t,zq,color=blue,lw=1.8,alpha=.8,label='查询（标准化）')
        a.plot(t,znorm(hit['values']),color=orange,lw=1.6,ls='--',label='匹配（标准化）')
        a.set_title(f"Top-{rank}  |  距离 {hit['distance']:.3f}  |  起点 {hit['start']}\n{hit['column']} · 设备 {hit.get('device','—')}",loc='left',fontsize=11)
        if rank%2==1:a.set_ylabel('标准化数值')
        if rank>=9:a.set_xlabel('相对采样点')
    for a in list(axes.flat)[len(hits):]:a.set_visible(False)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.966),ncol=2)
    fig.suptitle('Top-10 匹配形状对比',fontsize=19,y=.995)
    fig.text(.5,.009,'每段分别减去均值并除以标准差；距离越小，形状越接近。Top-2～10 不是原片段的完全复制。',ha='center',fontsize=11)
    fig.tight_layout(rect=(0,.025,1,.94));save(fig,'top10_shapes')

    sid=int(h['sid']);start=int(h['local_start']);series=idx.values(sid)
    a=max(0,start-1024);b=min(len(series),start+len(q)+1024);offset=int(idx.manifest['series'][sid].get('start',0))
    fig,ax=plt.subplots(figsize=(13,4.3))
    ax.plot(np.arange(a,b)+offset,series[a:b],color='#64748b',lw=1.2,label='来源长段（匹配前后各最多 1,024 点）')
    ax.axvspan(offset+start,offset+start+len(q),color=orange,alpha=.15)
    ax.plot(t+offset+start,v,color=orange,lw=2.2,label='定位到的 244 点')
    ax.set_title(f"长段中的精确位置：[{h['start']}, {h['start']+len(q)})",loc='left',fontsize=16)
    ax.set(xlabel='该设备序列中的采样点编号（0 起算，右端不含）',ylabel='原始数值');ax.ticklabel_format(axis='x',style='plain',useOffset=False)
    ax.legend(loc='best',fontsize=10);fig.tight_layout();save(fig,'source_context')
    idx.close()
    summary=dict(query_path=str(Path(args.query).resolve()),result_path=str(Path(args.result).resolve()),
        query_length=len(q),font=font,certified=result.get('certified'),best_max_absolute_error=err,
        hits=[{key:value for key,value in hit.items() if key!='values'} for hit in hits])
    (out/'details.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(out.resolve()),max_absolute_error=err,font=font,plots=['query_vs_best','top10_shapes','source_context']),ensure_ascii=False))

if __name__=='__main__':main()
