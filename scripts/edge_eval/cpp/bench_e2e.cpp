// End-to-end YOLOv8 counting pipeline in C++ for RV1126: stb JPEG load + bilinear
// letterbox + librknn_api NPU inference + optimized DFL decode + NMS, with per-stage
// timing.  Decode threshold-first: class sigmoid is checked before the (costly) DFL,
// so only surviving anchors are decoded.  Build: arm-linux-g++ -O3 -mfpu=neon.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <dirent.h>
#include <time.h>
#include "rknn_api.h"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#include "stb_image.h"

static const int NC = 2, REG = 16;
static int NET = 320;
static float CONF = 0.25f, IOU = 0.7f;
static int BOXCH = 64;      // 64 = raw DFL logits; 4 = NPU-fused distances
static int FUSED = 0;

static double now_ms(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1000.0 + t.tv_nsec/1e6; }
static inline float sigmoidf(float x){ return 1.f/(1.f+expf(-x)); }

struct Box{ float x1,y1,x2,y2,score; int cls; };

// bilinear letterbox RGB(uint8) -> NET*NET*3 uint8, pad 114
static void letterbox(const unsigned char*src,int W,int H,unsigned char*dst,float*scale,int*padx,int*pady){
  float r=std::min((float)NET/W,(float)NET/H);
  int nw=(int)(W*r+0.5f), nh=(int)(H*r+0.5f);
  int px=(NET-nw)/2, py=(NET-nh)/2; *scale=r; *padx=px; *pady=py;
  memset(dst,114,NET*NET*3);
  for(int y=0;y<nh;y++){
    float sy=(y+0.5f)/r-0.5f; int y0=(int)floorf(sy); float fy=sy-y0;
    if(y0<0){y0=0;fy=0;} if(y0>=H-1){y0=H-2<0?0:H-2;fy=1;}
    const unsigned char*r0=src+(size_t)y0*W*3, *r1=src+(size_t)(y0+ (y0<H-1))*W*3;
    unsigned char*o=dst+((size_t)(y+py)*NET+px)*3;
    for(int x=0;x<nw;x++){
      float sx=(x+0.5f)/r-0.5f; int x0=(int)floorf(sx); float fx=sx-x0;
      if(x0<0){x0=0;fx=0;} if(x0>=W-1){x0=W-2<0?0:W-2;fx=1;}
      int x1=x0+(x0<W-1);
      for(int c=0;c<3;c++){
        float a=r0[x0*3+c]*(1-fx)+r0[x1*3+c]*fx;
        float b=r1[x0*3+c]*(1-fx)+r1[x1*3+c]*fx;
        o[x*3+c]=(unsigned char)(a*(1-fy)+b*fy+0.5f);
      }
    }
  }
}

static float iou(const Box&a,const Box&b){
  float xx1=std::max(a.x1,b.x1),yy1=std::max(a.y1,b.y1),xx2=std::min(a.x2,b.x2),yy2=std::min(a.y2,b.y2);
  float w=std::max(0.f,xx2-xx1),h=std::max(0.f,yy2-yy1),inter=w*h;
  float ua=(a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-inter;
  return ua<=0?0:inter/ua;
}
static void nms(std::vector<Box>&v){
  std::sort(v.begin(),v.end(),[](const Box&a,const Box&b){return a.score>b.score;});
  std::vector<char> dead(v.size(),0); std::vector<Box> out;
  for(size_t i=0;i<v.size();i++){ if(dead[i])continue; out.push_back(v[i]);
    for(size_t j=i+1;j<v.size();j++) if(!dead[j]&&v[i].cls==v[j].cls&&iou(v[i],v[j])>IOU) dead[j]=1; }
  v.swap(out);
}

// decode one scale: box[64,H,W] NCHW, cls[NC,H,W] NCHW. Threshold cls first.
static void decode(const float*box,const float*cls,int H,int W,int stride,
                   float scale,int padx,int pady,std::vector<Box>&out){
  int HW=H*W;
  for(int y=0;y<H;y++)for(int x=0;x<W;x++){
    int idx=y*W+x;
    // class first (cheap): sigmoid over NC, find max
    int bc=0; float bl=cls[0*HW+idx];
    for(int c=1;c<NC;c++){ float v=cls[c*HW+idx]; if(v>bl){bl=v;bc=c;} }
    float score=sigmoidf(bl);
    if(score<CONF) continue;                      // skip background before decode
    float d[4];
    if(FUSED){                                    // distances already computed on NPU
      for(int s=0;s<4;s++) d[s]=box[s*HW+idx];
    } else {                                       // DFL: softmax over REG bins + expectation
      for(int s=0;s<4;s++){
        const float*p=box+((s*REG)*HW)+idx; float m=-1e9f;
        for(int b=0;b<REG;b++){ float v=p[b*HW]; if(v>m)m=v; }
        float sum=0,e[REG];
        for(int b=0;b<REG;b++){ e[b]=expf(p[b*HW]-m); sum+=e[b]; }
        float acc=0; for(int b=0;b<REG;b++) acc+=b*e[b];
        d[s]=acc/sum;
      }
    }
    float ax=x+0.5f, ay=y+0.5f;
    float x1=(ax-d[0])*stride, y1=(ay-d[1])*stride, x2=(ax+d[2])*stride, y2=(ay+d[3])*stride;
    x1=(x1-padx)/scale; y1=(y1-pady)/scale; x2=(x2-padx)/scale; y2=(y2-pady)/scale;
    Box b; b.x1=x1;b.y1=y1;b.x2=x2;b.y2=y2;b.score=score;b.cls=bc; out.push_back(b);
  }
}

int main(int argc,char**argv){
  if(argc<4){ fprintf(stderr,"usage: %s model.rknn imgsz imagesdir [N] [conf]\n",argv[0]); return 1; }
  std::string model=argv[1]; NET=atoi(argv[2]); std::string dir=argv[3];
  int N=argc>4?atoi(argv[4]):40; if(argc>5)CONF=atof(argv[5]);
  if(argc>6){ FUSED=atoi(argv[6]); BOXCH=FUSED?4:64; }

  FILE*f=fopen(model.c_str(),"rb"); fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
  std::vector<unsigned char> mb(sz); if(fread(mb.data(),1,sz,f)!=(size_t)sz){return 2;} fclose(f);
  rknn_context ctx; if(rknn_init(&ctx,mb.data(),sz,0)<0){fprintf(stderr,"rknn_init fail\n");return 3;}
  rknn_input_output_num ion; rknn_query(ctx,RKNN_QUERY_IN_OUT_NUM,&ion,sizeof ion);
  int on=ion.n_output;
  std::vector<rknn_tensor_attr> oa(on);
  for(int i=0;i<on;i++){ memset(&oa[i],0,sizeof(rknn_tensor_attr)); oa[i].index=i; rknn_query(ctx,RKNN_QUERY_OUTPUT_ATTR,&oa[i],sizeof(rknn_tensor_attr)); }

  // list images
  std::vector<std::string> files; DIR*dp=opendir(dir.c_str()); struct dirent*de;
  while((de=readdir(dp))){ std::string n=de->d_name; if(n.size()>4&&n.substr(n.size()-4)==".jpg") files.push_back(dir+"/"+n); }
  closedir(dp); std::sort(files.begin(),files.end()); if((int)files.size()>N) files.resize(N);

  // preload decoded RGB frames (isolate compute from disk; live capture replaces this)
  std::vector<std::vector<unsigned char>> frames; std::vector<int> Ws,Hs;
  for(auto&p:files){ int w,h,ch; unsigned char*d=stbi_load(p.c_str(),&w,&h,&ch,3);
    if(!d)continue; frames.emplace_back(d,d+(size_t)w*h*3); Ws.push_back(w); Hs.push_back(h); stbi_image_free(d); }
  int nf=frames.size();

  std::vector<unsigned char> lb(NET*NET*3);
  auto infer=[&](int i,double&tp,double&ti,double&to){
    float scale;int px,py; double a;
    a=now_ms(); letterbox(frames[i].data(),Ws[i],Hs[i],lb.data(),&scale,&px,&py); tp+=now_ms()-a;
    rknn_input in; memset(&in,0,sizeof in); in.index=0; in.type=RKNN_TENSOR_UINT8; in.size=NET*NET*3; in.fmt=RKNN_TENSOR_NHWC; in.buf=lb.data();
    a=now_ms(); rknn_inputs_set(ctx,1,&in); rknn_run(ctx,0);
    std::vector<rknn_output> outs(on); for(int k=0;k<on;k++){ memset(&outs[k],0,sizeof(rknn_output)); outs[k].want_float=1; outs[k].index=k; }
    rknn_outputs_get(ctx,on,outs.data(),0); ti+=now_ms()-a;
    a=now_ms();
    std::vector<Box> dets;
    // pair box(64ch) & cls(NC) by grid size
    for(int k=0;k<on;k++){
      int ne=oa[k].n_elems; // box output has BOXCH channels (64 raw / 4 fused)
      int Hb=(int)(sqrtf((float)ne/BOXCH)+0.5f);
      if(Hb*Hb*BOXCH==ne){ // box output; find matching cls (NC ch) with same H
        int H=Hb;
        for(int j=0;j<on;j++){ int ne2=oa[j].n_elems; if(ne2==NC*H*H){
          decode((float*)outs[k].buf,(float*)outs[j].buf,H,H,NET/H,scale,px,py,dets); break; } }
      }
    }
    nms(dets);
    to+=now_ms()-a;
    rknn_outputs_release(ctx,on,outs.data());
    return (int)dets.size();
  };
  double tp=0,ti=0,to=0; int warm=std::min(3,nf);
  for(int i=0;i<warm;i++){ double a=0,b=0,c=0; infer(i,a,b,c); }
  int total=0;
  for(int i=0;i<nf;i++) total+=infer(i,tp,ti,to);
  double pre=tp/nf,inf=ti/nf,post=to/nf,e2e=pre+inf+post;
  const char*mn=strrchr(model.c_str(),'/'); mn=mn?mn+1:model.c_str();
  printf("CPP %-16s conf=%.2f pre=%.1f inf=%.1f post=%.1f => %.1f ms  %.1f fps  (avg %d det)\n",
         mn,CONF,pre,inf,post,e2e,1000.0/e2e, nf?total/nf:0);
  rknn_destroy(ctx);
  return 0;
}
