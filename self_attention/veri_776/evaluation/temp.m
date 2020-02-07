
q = [1,4,7,10; 2,5,8,11; 3,6,9,12]
t = [1,4,7,10,11;2,4,8,10,12;3,4,9,10,13]

q = norm_code(q)
t = norm_code(t)

dist = 1 - t' * q

dist_2 = zeros(5, 4);

for i = 1:5
  for j = 1:4
    X1 = t(:,i);
    X2 = q(:,j);
    dist_2(i, j) = bhattacharyya(X1, X2);
  end
end

dist_2

function d = bhattacharyya(X1, X2)
  %check inputs and output
  error(nargchk(2,2,nargin));
  error(nargoutchk(0,1,nargout));

  [n,m]=size(X1);
  % check dimension 
  % assert(isequal(size(X2),[n m]),'Dimension of X1 and X2 mismatch.');
  assert(size(X2,2)==m,'Dimension of X1 and X2 mismatch.');

  mu1=mean(X1);
  C1=cov(X1);
  mu2=mean(X2);
  C2=cov(X2);
  C=(C1+C2)/2;
  dmu=(mu1-mu2)/chol(C);
  try
    d=0.125*dmu*dmu'+0.5*log(det(C/chol(C1*C2)));
  catch
    d=0.125*dmu*dmu'+0.5*log(abs(det(C/sqrtm(C1*C2))));
    warning('MATLAB:divideByZero','Data are almost linear dependent. The results may not be accurate.');
  end
end
