% COMPUTE VALUES FOR MNIST
tol = 5e-2 ;

[trX, trY, teX, teY] = load_usps(Ntr, Nte) ;
[trX,trY] = group_data(trX,trY) ;
[teX,teY] = group_data(teX,teY) ;
disp('Loading complete') ;

save('Y.mat','trY','teY') ;

N2 = ones(length(trY),1) ;
Nt2 = ones(length(teY),1) ;

NW = squeeze(sum(sum(trX,1),2)) ;
NtW = squeeze(sum(sum(teX,1),2)) ;

save('N2.mat','N2','-v7.3') ;
save('Nt2.mat','Nt2','-v7.3') ;
save('NW.mat','NW','-v7.3') ;
save('NtW.mat','NtW','-v7.3') ;

DW = wass_dists(trX,[],lambda,1,p_single,p_gpu,0,tol) ;
disp('1/4') ;
save('DW.mat','DW','-v7.3') ;
clear DW ;

D2 = l2_dists(trX,[],0,p_single) ;
disp('2/4') ;
save('D2.mat','D2','-v7.3') ;
clear D2 ;

DtW = wass_dists(trX,teX,lambda,1,p_single,p_gpu,0,tol) ;
disp('3/4') ;
save('DtW.mat','DtW','-v7.3') ;
clear DtW ;

Dt2 = l2_dists(trX,teX,0,p_single) ;
disp('4/4') ;
save('Dt2.mat','Dt2','-v7.3') ;
clear Dt2 ;

disp('Distances and norms computed') ;
