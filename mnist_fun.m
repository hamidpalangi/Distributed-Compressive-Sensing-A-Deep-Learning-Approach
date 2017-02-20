function mnists = mnist_fun(vec_in,n,k)
mnists = vec_in;
[temp111,sel] = sort(abs(mnists));
mnists( sel(1:n-k) ) = 0;
end