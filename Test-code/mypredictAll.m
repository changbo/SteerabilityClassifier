function sol = mypredictAll()
% k = 6;
k = 9;
%¡¡tag = 'nopartial-';
tag = '';

timefile=['predict-time',int2str(k),'-features-',tag,'.txt'];
% for ii=2:2    
%     myfun = @()mysvm(ii);
%     t = timeit(myfun);
%     dlmwrite(timefile,[ii, t],'-append','delimiter',' ','precision','%.4f');
% end
for ii=2:8
    tic;
    display(tic);
    mysvmpredict(ii);
    toc;
    display(toc);
    dlmwrite(timefile,[ii, toc],'-append','delimiter',' ','precision','%.4f');
end
end



