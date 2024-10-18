paramdf=data.frame()
inplane = c(2,2,1,1,1,1)
zplane  = c(2,4,1,2,3,4)
szs=data.frame( sz=rep(72,length(inplane)), x=inplane, y=inplane, z=zplane )
feats=data.frame( layer=c(6,6), feat=c("vgg","grader") )
# vgg layers = 4, 19
# grader layers 6, 25
# seg = 0, 1
ct=1
n=0
# for ( n in c(0,1) )
  for ( k in 1:nrow(feats) ) {
    for ( j in 1:nrow( szs ) ) {
      paramdf[ct,names(szs)]=szs[j,names(szs)]
      paramdf[ct,names(feats)]=feats[k,names(feats)]
      paramdf[ct,'seg']=n
      ct=ct+1
      }
    }
write.csv(paramdf,"train_params_small.csv",row.names=FALSE)

