function mysvm(m)

k = 9;
 tag =¡¡'';
%k = 6;
%tag = 'nopartial';

%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% posdatafilename=[int2str(m),'-positive-mixed','-5000','.txt'];
% negdatafilename=[int2str(m),'-negtive-mixed','-5000','.txt'];
% tunetagfilename=[int2str(m),'-tune-tag','-5000','.txt'];
% tunedfilename=[int2str(m),'-tuned-parameter','-5000','.txt'];

% posdatafilename=[int2str(m),'-positive-mixed-reduced','.txt'];
% negdatafilename=[int2str(m),'-negtive-mixed-reduced','.txt'];
% tunetagfilename=[int2str(m),'-tune-tag-reduced','.txt'];
% tunedfilename=[int2str(m),'-tuned-parameter-reduced','.txt'];

posdatafilename=[int2str(m),'-positive-mixed-',tag,int2str(k),'features','.txt'];
negdatafilename=[int2str(m),'-negtive-mixed-',tag,int2str(k),'features','.txt'];
tunetagfilename=[int2str(m),'-tune-tag-',tag,int2str(k),'features','.txt'];
tunedfilename=[int2str(m),'-tuned-parameter-',tag,int2str(k),'features','.txt'];

posdata = dlmread(posdatafilename);
negdata = dlmread(negdatafilename);

% normalize the data does not help.
% posdata = [posdata(:,1) zscore(posdata(:,2:end))];
% negdata = [negdata(:,1) zscore(negdata(:,2:end))];
% posdata = [posdata(:,1) rescaleData(posdata(:,2:end))];
% negdata = [negdata(:,1) rescaleData(negdata(:,2:end))];

possize = 1000;
negsize = 1000;

circsize=possize+negsize;

modelsize=circsize*3;
crosssize=circsize*4;

crossdata = [posdata(1:possize,:);negdata(1:negsize,:);posdata(possize+1:2*possize,:);negdata(negsize+1:2*negsize,:);posdata(2*possize+1:3*possize,:);negdata(2*negsize+1:3*negsize,:);posdata(3*possize+1:4*possize,:);negdata(3*negsize+1:4*negsize,:)];
testdata = [posdata(4*possize+1:5*possize,:); negdata(4*negsize+1:5*negsize,:)];

oldcrossrate=0;
myjj = 0; mykk = 0;
for jj = -10:15
    for kk = -5:10
%  for jj = 3:3
%      for kk = -2:-2
%   for jj=1:0.1:3
%       for kk = -1:0.1:1
        crossrate=0;
        cc=2^jj
        gg=2^kk
        cmd = ['-s 0 -t 2 -c ', num2str(cc), ' -g ', num2str(gg), ' -h ', num2str(0), ' -q'];
        crossdata = [posdata(1:possize,:);negdata(1:negsize,:);posdata(possize+1:2*possize,:);negdata(negsize+1:2*negsize,:);posdata(2*possize+1:3*possize,:);negdata(2*negsize+1:3*negsize,:);posdata(3*possize+1:4*possize,:);negdata(3*negsize+1:4*negsize,:)];
        for i = 1:4
            if i>1
                crossdata=circshift(crossdata, -circsize, 1);
            end 
            modeldata=crossdata(1:modelsize,:);
            tunedata=crossdata(modelsize+1:crosssize,:);
            modelclass = modeldata(:,1);
            modelvector = modeldata(:,2:end);
            mymodel=svmtrain(modelclass, modelvector,cmd);
            tuneclass = tunedata(:,1);
            tunevector = tunedata(:,2:end);
            v1=svmpredict(tuneclass,tunevector,mymodel);
            p1=tuneclass+v1;
            display('the success ratio for ML with m= 2 tag')
            crossrate = crossrate + sum(p1~=0)/length(p1);
        end
        crossrate=crossrate*0.25;
        dlmwrite(tunetagfilename,[jj kk crossrate],'-append','delimiter',' ','precision','%.15f');
        display(crossrate)
        if crossrate > oldcrossrate
           oldcrossrate=crossrate;
           myjj=jj;
           mykk=kk;
        end
    end
 end

 oldcrossrate=0;
 
 for jj = myjj-1:0.2:myjj+1
     for kk = mykk-1:0.2:mykk+1
%  for jj = myjj-1:1:myjj+1
%      for kk = mykk-1:1:mykk+1
%   for jj=1:0.1:3
%       for kk = -1:0.1:1
        crossrate=0;
        cc=2^jj
        gg=2^kk
        cmd = ['-s 0 -t 2 -c ', num2str(cc), ' -g ', num2str(gg), ' -h ', num2str(0), ' -q'];
        crossdata = [posdata(1:possize,:);negdata(1:negsize,:);posdata(possize+1:2*possize,:);negdata(negsize+1:2*negsize,:);posdata(2*possize+1:3*possize,:);negdata(2*negsize+1:3*negsize,:);posdata(3*possize+1:4*possize,:);negdata(3*negsize+1:4*negsize,:)];
        for i = 1:4
            if i>1
                crossdata=circshift(crossdata, -circsize, 1);
            end 
            modeldata=crossdata(1:modelsize,:);
            tunedata=crossdata(modelsize+1:crosssize,:);
            modelclass = modeldata(:,1);
            modelvector = modeldata(:,2:end);
            mymodel=svmtrain(modelclass, modelvector,cmd);
            tuneclass = tunedata(:,1);
            tunevector = tunedata(:,2:end);
            v1=svmpredict(tuneclass,tunevector,mymodel);
            p1=tuneclass+v1;
            display('the success ratio for ML with m= 2 tag')
            crossrate = crossrate + sum(p1~=0)/length(p1);
        end
        crossrate=crossrate*0.25;
        dlmwrite(tunetagfilename,[jj kk crossrate],'-append','delimiter',' ','precision','%.15f');
        display(crossrate)
        if crossrate > oldcrossrate
           oldcrossrate=crossrate;
           myjj=jj;
           mykk=kk;
        end
    end
end
 
    dlmwrite(tunedfilename,[myjj mykk oldcrossrate],'delimiter',' ','precision','%d');
    display('the best parameter');
    myjj
    mykk
end

