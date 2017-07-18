#ifndef RDP_HPP
#define RDP_HPP

#include <vector>
#include <algorithm>
#include <numeric>

template <typename T>
class RDP
{
public:

    static bool FarthestPoint(const T *px, const T *py, size_t np, double la, double lb, double lc, size_t &pidx, double &dist)
    {
        if(np<1)
            return false;

        pidx=0;
        dist=fabs(px[0]*la+py[0]*lb+lc);

        for(size_t i=1;i<np;i++)
        {
            double dt=fabs(px[i]*la+py[i]*lb+lc);
            if(dt>dist)
            {
                dist=dt;
                pidx=i;
            }
        }

        dist/=sqrt(la*la+lb*lb);

        return true;
    }
    static void LinePara(double &la, double &lb, double &lc, T x1, T y1, T x2, T y2)
    {
        la=y1-y2;
        lb=x2-x1;
        lc=x1*y2-x2*y1;
    }
    static void Approx(const std::vector<T> &x, const std::vector<T> &y, std::vector<size_t> &seg, size_t n, double distThres=0)
    {
        seg.assign(2,0);
        seg[1]=x.size()-1;

        std::vector<size_t> farseg;
        std::vector<double> fardist;



        {
            double lp[3];

            LinePara(lp[0],lp[1],lp[2],x[seg[0]],y[seg[0]],x[seg[1]],y[seg[1]]);

            size_t newidx;
            double newd;
            bool flg=FarthestPoint(&x[seg[0]+1],&y[seg[0]+1],seg[1]-seg[0]-1,lp[0],lp[1],lp[2],newidx,newd);

            farseg.push_back(newidx+seg[0]+1);
            fardist.push_back(newd);
            seg.insert(seg.begin()+1,newidx+seg[0]+1);
        }

        while(seg.size()<n+1)
        {
            for(size_t i=1;i<seg.size()-1;i++)
            {
                if(seg[i]==farseg[i-1])
                {
                    double lp[3];
                    LinePara(lp[0],lp[1],lp[2],x[seg[i-1]],y[seg[i-1]],x[seg[i]],y[seg[i]]);
                    size_t newidx[2]={x.size(),x.size()};
                    double newd[2]={0,0};
                    bool flg[2];
                    flg[0]=FarthestPoint(&x[seg[i-1]+1],&y[seg[i-1]+1],seg[i]-seg[i-1]-1,lp[0],lp[1],lp[2],newidx[0],newd[0]);

                    if(flg[0])
                    {
                        newidx[0]+=seg[i-1]+1;
                    }

                    LinePara(lp[0],lp[1],lp[2],x[seg[i]],y[seg[i]],x[seg[i+1]],y[seg[i+1]]);
                    flg[1]=FarthestPoint(&x[seg[i]+1],&y[seg[i]+1],seg[i+1]-seg[i]-1,lp[0],lp[1],lp[2],newidx[1],newd[1]);

                    if(flg[1])
                    {
                        newidx[1]+=seg[i]+1;
                    }

                    farseg.insert(farseg.begin()+i-1,newidx,newidx+2);
                    farseg.erase(farseg.begin()+i+1);

                    fardist.insert(fardist.begin()+i-1,newd,newd+2);
                    fardist.erase(fardist.begin()+i+1);

                    size_t faridx=std::max_element(fardist.begin(),fardist.end())-fardist.begin();

                    if(fardist[faridx]>distThres)
                    {
                        seg.insert(seg.begin()+faridx+1,farseg[faridx]);
                    }
                    else
                    {
                        return;
                    }

                    break;
                }
            }
        }

    }


    static bool FarthestPoint(const T *p, size_t np, const double *l, size_t &pidx, double &dist)
    {
        if(np<1)
            return false;
        std::vector<double> dt(np,0);
        for(size_t i=0;i<np;i++)
            dt[i]=fabs(std::inner_product(&p[2*i],&p[2*(i+1)],l,l[2]));
        pidx=std::max_element(dt.begin(),dt.end())-dt.begin();
        dist=dt[pidx]/sqrt(std::inner_product(l,&l[2],l,0));
        return true;
    }

    static size_t FarthestPoint(const T *p, size_t np, size_t index, double &dist)
    {
        std::vector<double> dt2(np,0);
        for(size_t i=0;i<np;i++)
        {
            T dx=p[2*i]-p[2*index];
            T dy=p[2*i+1]-p[2*index+1];
            dt2[i]=dx*dx+dy*dy;
        }
        size_t pidx=std::max_element(dt2.begin(),dt2.end())-dt2.begin();
        dist=sqrt(dt2[pidx]);
        return pidx;
    }




    static void LinePara(double *l, const T *p1, const T *p2)
    {
        l[0]=p1[1]-p2[1];
        l[1]=p2[0]-p1[0];
        l[2]=p1[0]*p2[1]-p2[0]*p1[1];
    }

    static bool FarthestPointA(const T *p, size_t startIndex, size_t endIndex, size_t &pidx, double &dist)
    {
        double lp[3];
        LinePara(lp,&p[2*startIndex],&p[2*endIndex]);
        size_t newidx;
        if(FarthestPoint(&p[2*(startIndex+1)],endIndex-startIndex-1,lp,newidx,dist))
        {
            pidx=newidx+startIndex+1;
            return true;
        }
        return false;
      }


    static bool Approx(const T *p, size_t np, std::vector<size_t> &segIndex, size_t nseg, double distThres=0)
    {
        if(p==NULL)
            return false;

        if(np<3)
            return false;

        segIndex.assign(2,0);
        segIndex[1]=np-1;
        if(nseg<2)
            return true;

        std::vector<size_t> farseg;
        std::vector<double> fardist;

        {
            size_t newidx;
            double newd;
            bool flg=FarthestPointA(p,segIndex[0],segIndex[1],newidx,newd);

            farseg.push_back(newidx);
            fardist.push_back(newd);
            segIndex.insert(segIndex.begin()+1,newidx);
        }

        while(segIndex.size()<nseg+1)
        {
            for(size_t i=1;i<segIndex.size()-1;i++)
            {
                if(segIndex[i]==farseg[i-1])
                {
                    size_t newidx[2]={np,np};
                    double newd[2]={0,0};
                    bool flg[2];

                    flg[0]=FarthestPointA(p,segIndex[i-1],segIndex[i],newidx[0],newd[0]);
                    flg[1]=FarthestPointA(p,segIndex[i],segIndex[i+1],newidx[1],newd[1]);

                    farseg.insert(farseg.begin()+i-1,newidx,newidx+2);
                    farseg.erase(farseg.begin()+i+1);

                    fardist.insert(fardist.begin()+i-1,newd,newd+2);
                    fardist.erase(fardist.begin()+i+1);

                    size_t faridx=std::max_element(fardist.begin(),fardist.end())-fardist.begin();

                    if(fardist[faridx]<distThres)
                    {
                        return true;
                    }
                    segIndex.insert(segIndex.begin()+faridx+1,farseg[faridx]);
                    break;
                }
            }
        }
        return true;

    }




    static bool ApproxClose(const T *p, size_t np, std::vector<size_t> &segIndex, size_t nseg, double distThres=0)
    {
        if(p==NULL)
            return false;

        if(np<3)
            return false;

        double d1;
        size_t i1=FarthestPoint(p,np,0,d1);

        T *newp=new T[2*(np+1)];
         memcpy(newp,&p[2*i1],2*(np-i1)*sizeof(T));
         memcpy(&newp[2*(np-i1)],p,2*(i1+1)*sizeof(T));

         segIndex.assign(2,0);
         segIndex[1]=np;


         std::vector<size_t> farseg;
         std::vector<double> fardist;

         {
             size_t i2=FarthestPoint(newp,np,0,d1);
             farseg.push_back(i2);
             fardist.push_back(d1);
             segIndex.insert(segIndex.begin()+1,i2);
         }

         while(segIndex.size()<nseg+1)
         {
             for(size_t i=1;i<segIndex.size()-1;i++)
             {
                 if(segIndex[i]==farseg[i-1])
                 {
                     size_t newidx[2]={np+1,np+1};
                     double newd[2]={0,0};
                     bool flg[2];

                     flg[0]=FarthestPointA(newp,segIndex[i-1],segIndex[i],newidx[0],newd[0]);
                     flg[1]=FarthestPointA(newp,segIndex[i],segIndex[i+1],newidx[1],newd[1]);

                     farseg.insert(farseg.begin()+i-1,newidx,newidx+2);
                     farseg.erase(farseg.begin()+i+1);

                     fardist.insert(fardist.begin()+i-1,newd,newd+2);
                     fardist.erase(fardist.begin()+i+1);

                     size_t faridx=std::max_element(fardist.begin(),fardist.end())-fardist.begin();

                     if(fardist[faridx]>distThres)
                     {
                         segIndex.insert(segIndex.begin()+faridx+1,farseg[faridx]);
                     }
                     else
                     {
                         segIndex.pop_back();
                         delete newp;

                         for(size_t j=0;j<segIndex.size();j++)
                             segIndex[j]=(segIndex[j]+i1)%np;

                         return true;
                     }

                     break;
                 }
             }
         }

         segIndex.pop_back();
         delete newp;

         for(size_t j=0;j<segIndex.size();j++)
             segIndex[j]=(segIndex[j]+i1)%np;

         return true;

    }



};
#endif // RDP_HPP
