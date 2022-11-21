paramdf=data.frame()
szs=data.frame( sz=rep(72,6), x=c(1,1,1,2,2,4), y=c(1,1,1,2,2,4), z=c(2,4,6,2,4,4) )
feats=data.frame( layer=c(6,6,6,25,21,21), feat=c("vgg","vggrandom","grader","grader","vgg","vggrandom") )
# vgg layers = 4, 19
# grader layers 6, 25
# seg = 0, 1
ct=1
for ( n in c(0,1) )
  for ( k in 1:nrow(feats) ) {
    for ( j in 1:nrow( szs ) ) {
      paramdf[ct,names(szs)]=szs[j,names(szs)]
      paramdf[ct,names(feats)]=feats[k,names(feats)]
      paramdf[ct,'seg']=n
      ct=ct+1
      }
    }
write.csv(paramdf,"train_params.csv",row.names=FALSE)
