/* Optional bounded candidate heap. Distances are still computed by NumPy using
   the original float64 arithmetic. No fast-math, all-distance cache or model. */
#include <stdint.h>
#include <math.h>
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif
static int worse(double d,int64_t s,int64_t p,double e,int64_t t,int64_t r){
    return d>e || (d==e && (s>t || (s==t && p>r)));
}
API void candidate_add(const double*values,const int64_t*starts,int64_t n,int64_t sid,
                      int64_t capacity,double*ds,int64_t*ss,int64_t*ps,int64_t*count){
    for(int64_t j=0;j<n;j++){
        double d=values[j];int64_t p=starts[j],i;
        if(!isfinite(d))continue;
        if(*count<capacity){
            i=(*count)++;
            while(i>0){int64_t parent=(i-1)/2;
                if(!worse(d,sid,p,ds[parent],ss[parent],ps[parent]))break;
                ds[i]=ds[parent];ss[i]=ss[parent];ps[i]=ps[parent];i=parent;
            }
        }else{
            if(!worse(ds[0],ss[0],ps[0],d,sid,p))continue;
            i=0;
            while(2*i+1<capacity){int64_t child=2*i+1;
                if(child+1<capacity && worse(ds[child+1],ss[child+1],ps[child+1],ds[child],ss[child],ps[child]))child++;
                if(!worse(ds[child],ss[child],ps[child],d,sid,p))break;
                ds[i]=ds[child];ss[i]=ss[child];ps[i]=ps[child];i=child;
            }
        }
        ds[i]=d;ss[i]=sid;ps[i]=p;
    }
}
