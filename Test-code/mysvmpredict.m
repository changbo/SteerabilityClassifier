function sol = mysvmpredict(m)

k = 9;
%k = 6;
tag = 'nopartial';
tag = '';
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%     posdatafilename=[int2str(m),'-positive-mixed','-5000','.txt'];
%     negdatafilename=[int2str(m),'-negtive-mixed','-5000','.txt'];
%     wernerfilename=[int2str(m),'-Werner-test','-5000','.txt'];
%     wernertagfilename=[int2str(m),'-Werner-tag','-5000','.txt'];
%     

    %----------------read the training and test data--------------

    posdatafilename=[int2str(m),'-positive-mixed-',tag, int2str(k),'features','.txt'];
    negdatafilename=[int2str(m),'-negtive-mixed-',tag, int2str(k),'features','.txt'];
    testfilename=[int2str(m),'-test-',tag, int2str(k),'features','.txt'];
    
    %----------------read the data of m=8---------------
    eightposfilename=[int2str(8),'-positive-mixed-',tag, int2str(k),'features','.txt'];
    eightnegfilename=[int2str(8),'-negtive-mixed-',tag, int2str(k),'features','.txt'];
    eightposdata=dlmread(eightposfilename);
    eightnegdata=dlmread(eightnegfilename);
    eighttestdata = [eightposdata(4000+1:5000,:); eightnegdata(4000+1:5000,:)]; 
    
    %--------------------werner----------------------------
    wernerfilename=[int2str(m),'-Werner-test-',tag, int2str(k),'features','.txt'];
   
    
    
    %----------------get the training and test data of m---------
    posdata = dlmread(posdatafilename);
    negdata = dlmread(negdatafilename);
    possize = 1000;
    negsize = 1000;
    modeldata = [posdata(1:possize,:);negdata(1:negsize,:);posdata(possize+1:2*possize,:);negdata(negsize+1:2*negsize,:);posdata(2*possize+1:3*possize,:);negdata(2*negsize+1:3*negsize,:);posdata(3*possize+1:4*possize,:);negdata(3*negsize+1:4*negsize,:)];
    testdata = [posdata(4*possize+1:5*possize,:); negdata(4*negsize+1:5*negsize,:)];
    
    %---------------get the learning parameter----------------------
    tunedfilename=[int2str(m),'-tuned-parameter-',tag, int2str(k),'features','.txt'];
    mypara=dlmread(tunedfilename);
    cc = mypara(1);
    gg = mypara(2);
    crossrate = mypara(3);

    display('the best tuned parameter');
    display(cc);
    display(gg);

    % cc = 9.4;
    % gg = -0.2;

    mycc = 2^(cc);
    mygg = 2^(gg);

    
    %---------------start training------------------
    cmd = ['-s 0 -t 2 -c ', num2str(mycc), ' -g ', num2str(mygg), ' -h ', num2str(0), ' -q'];    
    modelclass = modeldata(:,1);
    modelvector = modeldata(:,2:end);
    mymodel=svmtrain(modelclass, modelvector,cmd);
    
    %--------------test data of m--------------
    
    testclass = testdata(:,1);
    testvector = testdata(:,2:end);
    [v2, accuracy, decision]=svmpredict(testclass,testvector,mymodel);
    p2=testclass+v2;
    display('the success ratio for ML for test')
    testrate = sum(p2~=0)/length(p2);
    display(testrate)
   

    fileID = fopen(testfilename,'w');
    fprintf(fileID,'%4s %4s\r\n','positiveAccuracy ', 'negtiveAccuracy ');
    fclose(fileID);

    posAccuracy=sum(v2(1:1000)>=0)/(1000);
    negAccuracy=sum(v2(1000+1:1000+1000)<0)/(1000);
    display(posAccuracy);
    display(negAccuracy);
    
    dlmwrite(testfilename,[posAccuracy, negAccuracy],'-append','delimiter',' ','precision','%.4f');
     
    %--------------test data of m=8--------------
    
    testclass = eighttestdata(:,1);
    testvector = eighttestdata(:,2:end);
    v2=svmpredict(testclass,testvector,mymodel);
    p2=testclass+v2;
    display('the success ratio for ML for test')
    eighttestrate = sum(p2~=0)/length(p2);
    display(eighttestrate)

    fileID = fopen(testfilename,'a');
    fprintf(fileID,'%4s %4s\r\n','eight-positiveAccuracy', 'eight-negtiveAccuracy');
    fclose(fileID);

    posAccuracy=sum(v2(1:1000)>=0)/(1000);
    negAccuracy=sum(v2(1000+1:1000+1000)<0)/(1000);
    
    dlmwrite(testfilename,[posAccuracy, negAccuracy],'-append','delimiter',' ','precision','%.4f');
    
    
     %-----------test result----------------------------------------
    fileID = fopen(testfilename,'a');
    fprintf(fileID,'%4s %4s %4s\r\n','crossrate', 'testrate', 'eighttestrate');
    fclose(fileID);
    
    dlmwrite(testfilename,[crossrate, testrate, eighttestrate],'-append','delimiter',' ','precision','%.4f');
    
    %----------------------------------------------------------------------------------------------
    
    
    
for nn=1:4
    
%     wernerdatafilename=['Werner-tag-9features-',int2str(nn),'.txt'];  
%     wernertagfilename=[int2str(m),'-Werner-tag-9features-',int2str(nn),'.txt'];
    wernerdatafilename=['Werner-tag-',tag, int2str(k),'features-',int2str(nn),'.txt'];  
    wernertagfilename=[int2str(m),'-Werner-tag-',tag, int2str(k),'features-',int2str(nn),'.txt'];

     
    wernerdata =  dlmread(wernerdatafilename);
    
    wernerdata = wernerdata(1:10000,:);
    wernerclass = wernerdata(:,1);
    wernervector = wernerdata(:,2:end);
    
    v3=svmpredict(wernerclass,wernervector,mymodel);
   
%     fileID = fopen(wernertagfilename,'w');
%     fprintf(fileID,'%4s\r\n','learned tag');
%     fclose(fileID);
%     
%     for jj=1:size(v3,1)
%         dlmwrite(wernertagfilename,[v3(jj)],'-append','delimiter',' ','precision','%d');
%     end
    
    p3=wernerclass+v3;
    display('the success ratio for ML for Werner')
    wernerrate=sum(p3~=0)/length(p3);
    display(wernerrate)
    
    fileID = fopen(wernerfilename,'a');
    fprintf(fileID,'%d %4s %4s\r\n', nn, 'positiveAccuracy', 'negtiveAccuracy');
    fclose(fileID);

    posAccuracy=sum(v3(1:5000)>=0)/(5000);
    negAccuracy=sum(v3(5001:10000)<0)/(5000);
    dlmwrite(wernerfilename,[posAccuracy, negAccuracy],'-append','delimiter',' ','precision','%.4f');
    
    
    u = linspace(0,1,10000+1);
    
    myu = 0;
    for jj=1:size(v3,1)
        if (v3(jj)<0) & (sum(v3(jj+1:size(v3,1))>= 0)==0)
           myu = u(jj);
           break;
        end
    end
    fileID = fopen(wernerfilename,'a');
    fprintf(fileID,'%d %4s %4s %4s\r\n',nn, 'wernerrate', 'firstNegtive', 'uvalue');
    fclose(fileID);
    %dlmwrite(wernerfilename,['crossrate testrate wernerrate firstNegtive uvalue'],'-append','delimiter',' ','precision','%.4f');
    dlmwrite(wernerfilename,[wernerrate, jj, myu],'-append','delimiter',' ','precision','%.4f');
    
end
    
    
    fileID = fopen(wernerfilename,'a');
    fprintf(fileID,'\n%4s %4s %4s\r\n','used parameter', 'cc', 'gg');
    fclose(fileID);
    
    dlmwrite(wernerfilename,[cc, gg],'-append','delimiter',' ','precision','%.4f');
  
   
    
end

