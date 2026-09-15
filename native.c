#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#define API __declspec(dllexport)
#else
#define API
#endif
#define PI 3.14159265358979323846
#define EPS 1e-10

static double moments(const double *x,int m,double *mean){
    double center=x[0],sum=0,v=0;
    for(int j=0;j<m;j++)sum+=x[j]-center;
    *mean=sum/m;
    for(int j=0;j<m;j++){double a=(x[j]-center)-*mean;v+=a*a;}
    return sqrt(v/m);
}

API void encode(const double*x,int64_t n,int m,const int64_t*freq,int nf,int npaa,
                const double*bins,uint8_t*codes,uint8_t*flat,double*debug){
    int d=2*nf+npaa;int64_t count=n-m+1;
    double *prefix=malloc((n+1)*sizeof(double));
    double *co=malloc(nf*m*sizeof(double)),*si=malloc(nf*m*sizeof(double));
    double *re=calloc(nf,sizeof(double)),*im=calloc(nf,sizeof(double));
    double *cr=malloc(nf*sizeof(double)),*sr=malloc(nf*sizeof(double));
    double *values=malloc(d*sizeof(double));
    if(!prefix||!co||!si||!re||!im||!cr||!sr||!values)abort();
    double center=x[0],mean=0,m2=0,factor=sqrt(2.0/m);
    prefix[0]=0;for(int64_t i=0;i<n;i++)prefix[i+1]=prefix[i]+(x[i]-center);
    for(int f=0;f<nf;f++){
        double w=2*PI*freq[f]/m;cr[f]=cos(w);sr[f]=sin(w);
        for(int j=0;j<m;j++){co[f*m+j]=cos(w*j);si[f*m+j]=-sin(w*j);}
    }
    int changes=0;for(int j=1;j<m;j++)changes+=(x[j]!=x[j-1]);
    for(int64_t s=0;s<count;s++){
        if(s%256==0){
            mean=0;for(int j=0;j<m;j++)mean+=x[s+j]-center;mean/=m;
            m2=0;for(int j=0;j<m;j++){double z=(x[s+j]-center)-mean;m2+=z*z;}
            for(int f=0;f<nf;f++){
                re[f]=im[f]=0;
                for(int j=0;j<m;j++){double z=(x[s+j]-center)-mean;re[f]+=z*co[f*m+j];im[f]+=z*si[f*m+j];}
            }
        }
        double sd=changes?sqrt(fmax(0,m2/m)):0;
        int robust=changes && (m2/m<1e-8 || sd<1e-7*fabs(mean));
        double localmean=0;
        if(robust){
            sd=moments(x+s,m,&localmean);mean=localmean+(x[s]-center);m2=sd*sd*m;
            for(int f=0;f<nf;f++){
                re[f]=im[f]=0;
                for(int j=0;j<m;j++){double z=(x[s+j]-x[s])-localmean;re[f]+=z*co[f*m+j];im[f]+=z*si[f*m+j];}
            }
        }
        flat[s]=(sd<=EPS);
        double denom=sd>EPS?sd:1;
        for(int f=0;f<nf;f++){
            values[2*f]=flat[s]?0:factor*re[f]/denom;
            values[2*f+1]=flat[s]?0:factor*im[f]/denom;
        }
        for(int j=0;j<npaa;j++){
            int a=j*m/npaa,b=(j+1)*m/npaa;
            if(robust){double sum=0;for(int k=a;k<b;k++)sum+=(x[s+k]-x[s])-localmean;
                values[2*nf+j]=flat[s]?0:sum/sqrt((double)(b-a))/denom;
            }else values[2*nf+j]=flat[s]?0:((prefix[s+b]-prefix[s+a])/(b-a)-mean)*sqrt((double)(b-a))/denom;
        }
        for(int j=0;j<d;j++){
            int lo=0,hi=255;const double *bp=bins+j*255;
            while(lo<hi){int mid=(lo+hi)/2;if(values[j]<bp[mid])hi=mid;else lo=mid+1;}
            codes[s*d+j]=(uint8_t)lo;
            if(debug)debug[s*d+j]=values[j];
        }
        if(s+1<count){
            double old=x[s]-center,in=x[s+m]-center,diff=in-old,newmean=mean+diff/m;
            m2+=diff*(in-newmean+old-mean);mean=newmean;
            for(int f=0;f<nf;f++){
                double a=re[f]+diff,b=im[f];re[f]=a*cr[f]-b*sr[f];im[f]=a*sr[f]+b*cr[f];
            }
            changes-=(x[s+1]!=x[s]);changes+=(x[s+m]!=x[s+m-1]);
        }
    }
    free(prefix);free(co);free(si);free(re);free(im);free(cr);free(sr);free(values);
}

API void exact_ed(const double*x,const int64_t*starts,int64_t n,int m,const double*q,
                  const int64_t*order,double cutoff,double*out){
    for(int64_t i=0;i<n;i++){
        const double*w=x+starts[i];double mean,sd=moments(w,m,&mean),distance=0;
        for(int j=0;j<m;j++){
            int k=(int)order[j];double z=sd>EPS?((w[k]-w[0])-mean)/sd:0;
            double delta=z-q[k];distance+=delta*delta;
            if(distance>cutoff+1e-7)break;
        }
        out[i]=distance<1e-20?0:distance;
    }
}

API void exact_dtw(const double*x,const int64_t*starts,int64_t n,int m,const double*q,
                   const double*lower,const double*upper,int radius,double cutoff,
                   double*out,int64_t*stats){
    double *z=malloc(m*sizeof(double)),*prev=malloc((m+1)*sizeof(double)),*curr=malloc((m+1)*sizeof(double));
    if(!z||!prev||!curr)abort();
    for(int64_t s=0;s<n;s++){
        const double*w=x+starts[s];double mean,sd=moments(w,m,&mean),lb=0;
        if(sd<=EPS){double energy=0;for(int j=0;j<m;j++)energy+=q[j]*q[j];out[s]=energy;stats[0]++;continue;}
        for(int j=0;j<m;j++){
            z[j]=sd>EPS?((w[j]-w[0])-mean)/sd:0;
            double a=lower[j]-z[j],b=z[j]-upper[j];a=a>0?a:0;if(b>a)a=b;lb+=a*a;
        }
        stats[0]++;
        if(lb>cutoff+1e-7){out[s]=INFINITY;continue;}
        stats[1]++;
        for(int j=0;j<=m;j++){prev[j]=INFINITY;curr[j]=INFINITY;}prev[0]=0;
        double result=INFINITY;
        for(int i=1;i<=m;i++){
            double rowmin=INFINITY;int a=i-radius>1?i-radius:1,b=i+radius<m?i+radius:m;
            curr[a-1]=INFINITY;if(b<m)curr[b+1]=INFINITY;
            for(int j=a;j<=b;j++){
                double delta=z[i-1]-q[j-1];
                double best=prev[j-1]<prev[j]?prev[j-1]:prev[j];if(curr[j-1]<best)best=curr[j-1];
                curr[j]=delta*delta+best;if(curr[j]<rowmin)rowmin=curr[j];
            }
            double*tmp=prev;prev=curr;curr=tmp;
            if(rowmin>cutoff+1e-7)break;
            if(i==m)result=prev[m];
        }
        out[s]=result<1e-20?0:result;
    }
    free(z);free(prev);free(curr);
}

API void local_stats(const double*x,int64_t n,int m,double*mean,double*sd){
    for(int64_t s=0;s<n-m+1;s++)sd[s]=moments(x+s,m,mean+s);
}

API void exact_ed_cached(const double*x,const int64_t*starts,int64_t n,int m,const double*q,
                        const int64_t*order,const double*mean,const double*sd,double cutoff,double*out){
    double energy=0;for(int j=0;j<m;j++)energy+=q[j]*q[j];
    for(int64_t i=0;i<n;i++){
        int64_t s=starts[i];const double*w=x+s;double distance=0;
        if(sd[s]<=EPS){out[i]=energy<1e-20?0:energy;continue;}
        for(int j=0;j<m;j++){
            int k=(int)order[j];double z=((w[k]-w[0])-mean[s])/sd[s];
            double delta=z-q[k];distance+=delta*delta;
            if(distance>cutoff+1e-7)break;
        }
        out[i]=distance<1e-20?0:distance;
    }
}

API void bounds(const uint8_t*lo,const uint8_t*hi,int64_t n,int d,int nsfa,
                const double*left,const double*right,const double*ql,const double*qu,
                int dtw,double*out){
    for(int64_t i=0;i<n;i++){
        double a=0,b=0;
        for(int j=0;j<d;j++){
            if(dtw && j<nsfa)continue;
            double l=left[j*256+lo[i*d+j]]-1e-4;
            double u=right[j*256+hi[i*d+j]]+1e-4;
            double delta=l-qu[j],other=ql[j]-u;delta=delta>0?delta:0;if(other>delta)delta=other;
            if(j<nsfa)a+=delta*delta;else b+=delta*delta;
        }
        out[i]=a>b?a:b;
    }
}

/* Query-specific symbol distances: early abandon a point only by a safe bound. */
API int64_t filter_codes(const uint8_t*codes,int64_t n,int d,int nsfa,
                        const double*lookup,double cutoff,int64_t*out){
    int64_t count=0;
    for(int64_t i=0;i<n;i++){
        double sum=0;int reject=0;
        for(int j=0;j<nsfa;j++){
            sum+=lookup[j*256+codes[i*d+j]];
            if(sum>cutoff+1e-7){reject=1;break;}
        }
        if(reject)continue;
        sum=0;
        for(int j=nsfa;j<d;j++){
            sum+=lookup[j*256+codes[i*d+j]];
            if(sum>cutoff+1e-7){reject=1;break;}
        }
        if(!reject)out[count++]=i;
    }
    return count;
}

typedef struct {double bound;int64_t node;} Item;
static void hp_push(Item*h,int64_t*size,Item v){
    int64_t p=(*size)++;
    while(p){int64_t parent=(p-1)/2;if(h[parent].bound<=v.bound)break;h[p]=h[parent];p=parent;}h[p]=v;
}
static Item hp_pop(Item*h,int64_t*size){
    Item result=h[0],v=h[--(*size)];int64_t p=0;
    while(2*p+1<*size){int64_t c=2*p+1;if(c+1<*size && h[c+1].bound<h[c].bound)c++;
        if(v.bound<=h[c].bound)break;h[p]=h[c];p=c;}if(*size)h[p]=v;return result;
}
static double millis(void){
#ifdef _WIN32
    LARGE_INTEGER t,f;QueryPerformanceCounter(&t);QueryPerformanceFrequency(&f);return 1000.*t.QuadPart/f.QuadPart;
#else
    struct timespec t;timespec_get(&t,TIME_UTC);return t.tv_sec*1000.+t.tv_nsec/1e6;
#endif
}
static void insert_best(double distance,int64_t sid,int64_t pos,int k,double*ds,int64_t*ss,int64_t*pp,int64_t*count){
    if(!isfinite(distance))return;
    for(int j=0;j<*count;j++)if(ss[j]==sid && pp[j]==pos)return;
    if(*count==k && distance>=ds[k-1])return;
    int p=(*count<k)?(int)(*count)++:k-1;
    while(p>0 && ds[p-1]>distance){ds[p]=ds[p-1];ss[p]=ss[p-1];pp[p]=pp[p-1];p--;}
    ds[p]=distance;ss[p]=sid;pp[p]=pos;
}

/* config: nodes, records, dimensions, SFA dimensions, window, metric, radius, k.
   records are packed (int64 sid,start,end,uint8 flat), 25 bytes each. */
API int search_tree(const double*raw,const int64_t*offsets,const uint8_t*lo,const uint8_t*hi,
    const int64_t*children,const int64_t*spans,const uint8_t*codes,const uint8_t*records,
    const int64_t*config,const double*left,const double*right,const double*ql,const double*qu,
    const double*q,const int64_t*point_order,const double*envlo,const double*envhi,
    double*best,int64_t*best_sid,int64_t*best_pos,int64_t*best_count,int64_t*constant_count,
    double budget_ms,int64_t*stats,double*frontier){
    int64_t nn=config[0];int d=(int)config[2],nsfa=(int)config[3],m=(int)config[4];
    int metric=(int)config[5],radius=(int)config[6],k=(int)config[7];
    Item*heap=malloc((nn+2)*sizeof(Item));if(!heap)return -1;
    int64_t size=0;double root,started=millis(),energy=0;
    for(int j=0;j<m;j++)energy+=q[j]*q[j];
    bounds(lo,hi,1,d,nsfa,left,right,ql,qu,metric,&root);hp_push(heap,&size,(Item){root,0});
    int stopped=0;*frontier=INFINITY;
    while(size){
        Item item=hp_pop(heap,&size);int64_t node=item.node;
        double cutoff=*best_count==k?best[k-1]:INFINITY;
        if(item.bound>cutoff+1e-7)break;
        if(cutoff==0)break;
        if(budget_ms>=0 && (stats[0]%64==0) && millis()-started>=budget_ms){stopped=1;*frontier=item.bound;break;}
        stats[0]++;
        if(children[node*2]>=0){
            for(int j=0;j<2;j++){
                int64_t child=children[node*2+j];double lb;
                bounds(lo+child*d,hi+child*d,1,d,nsfa,left,right,ql,qu,metric,&lb);
                if(lb<=cutoff+1e-7)hp_push(heap,&size,(Item){lb,child});
            }
            continue;
        }
        stats[1]++;
        for(int64_t r=spans[node*2];r<spans[node*2+1];r++){
            if(budget_ms>=0 && r%128==0 && millis()-started>=budget_ms){stopped=1;*frontier=item.bound;goto done;}
            if(records[r*25+24] && *constant_count>=k){
                int64_t a,b;memcpy(&a,records+r*25+8,8);memcpy(&b,records+r*25+16,8);stats[5]+=b-a;stats[2]++;continue;
            }
            cutoff=*best_count==k?best[k-1]:INFINITY;
            double lb;bounds(codes+r*d,codes+r*d,1,d,nsfa,left,right,ql,qu,metric,&lb);stats[2]++;
            if(lb>cutoff+1e-7)continue;
            int64_t sid,a,b;memcpy(&sid,records+r*25,8);memcpy(&a,records+r*25+8,8);memcpy(&b,records+r*25+16,8);
            if(records[r*25+24]){
                stats[5]+=b-a;
                while(a<b && *constant_count<k){insert_best(energy,sid,a++,k,best,best_sid,best_pos,best_count);(*constant_count)++;}
                continue;
            }
            for(int64_t pos=a;pos<b;pos++){
                if(budget_ms>=0 && stats[3]%128==0 && millis()-started>=budget_ms){stopped=1;*frontier=item.bound;goto done;}
                cutoff=*best_count==k?best[k-1]:INFINITY;
                double distance;stats[3]++;
                if(metric){
                    int64_t c[2]={0,0};exact_dtw(raw+offsets[sid],&pos,1,m,q,envlo,envhi,radius,cutoff,&distance,c);stats[4]+=c[1];
                }else exact_ed(raw+offsets[sid],&pos,1,m,q,point_order,cutoff,&distance);
                if(distance<=cutoff)insert_best(distance,sid,pos,k,best,best_sid,best_pos,best_count);
            }
        }
    }
done:
    if(size)*frontier=fmin(*frontier,heap[0].bound);
    free(heap);return stopped;
}

API void window_stats(const double*x,int64_t n,int m,double*mean,double*sd){
    for(int64_t s=0;s<n-m+1;s++){
        double mu;sd[s]=moments(x+s,m,&mu);mean[s]=(x[s]-x[0])+mu;
    }
}
