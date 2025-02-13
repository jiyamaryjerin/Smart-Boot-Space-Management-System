def func(a):
l.append(img1(a))


a=pic1
b=pic2
c=pic3
#front view- fw/4,fh/4= img1
#side view- sd/4= img2
#boot- width=max(width_l), height=median= img3
l=[]
l.append(img1(a))
l.append(img2(b))
l.append(boot(c))
dim={}
dim.update({'fw':l[0][0],'fh':l[0][1],'sd':l[1],'bw':l[2][0],'bh':l[2][1]})
print(dim)
dim=imgs()
df= pd.DataFrame(dim)
print(df)
df.to_csv('C:/Users/rbpav/OneDrive/Desktop/hackathon/sample2.csv')





a=pic1
b=pic2
c=pic3
#front view- fw/4,fh/4= img1
#side view- sd/4= img2
#boot- width=max(width_l), height=median= img3
l=[]



l.append(img1(a))
l.append(img2(b))
l.append(boot(c))