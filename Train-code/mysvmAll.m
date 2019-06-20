function sol = mysvmAll()

k=9
%k = 6;
%tag = 'nopartial-';
tag = '';
timefile=[int2str(k),'-features-',tag,'training-time','.txt'];
% for ii=2:2    
%     myfun = @()mysvm(ii);
%     t = timeit(myfun);
%     dlmwrite(timefile,[ii, t],'-append','delimiter',' ','precision','%.4f');
% end
for ii=2:8
    tic;
    display(tic);
    mysvm(ii);
    toc;
    display(toc);
    dlmwrite(timefile,[ii, toc],'-append','delimiter',' ','precision','%.4f');
end
end



