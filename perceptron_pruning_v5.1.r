library(MASS)
library(stats)

# ——— Model definitions ——— 

phi <- function(x) x; 
#phi <- function(x) tanh(x);

# Loss L(w) and energy E(w|h)
L <- function(w) {
  preds <- phi(X %*% w)
  sum((y - preds)^2) / 2
}

E <- function(w, h) {
  L(w * h) + (eta * sum(w^2) / 2)
}

#commented code
if (FALSE) {
  # Gradient ∂E/∂w for \phi(x)=\tanh(x)
dE <- function(w, h) {
  pr_w <- w * h
  preds <- phi(X %*% pr_w)
  err   <- preds - y
  grad_loss <- t(X) %*% (err * (1 - preds^2)) * h
  grad_reg  <- eta * w
  as.vector(grad_loss + grad_reg)
}
}

# Gradient ∂E/∂w for \phi(x)=x
dE <- function(w, h) {
  pr_w   <- w * h
  preds  <- X %*% pr_w        # since phi is identity
  err    <- preds - y         # vector of residuals
  
  # ∂L/∂w = Xᵀ err * h
  grad_loss <- t(X) %*% err * h
  
  # ∂(η/2||w||²)/∂w = η w
  grad_reg  <- eta * w
  
  # total gradient & freeze pruned coords
  grad_total <- as.vector(grad_loss + grad_reg)

}

# ——— Adam optimizer helpers ———

init_adam <- function(N) {
  list(m = numeric(N), v = numeric(N), t = 0L)
}

adam_step <- function(w, gw, state,
                      lr   = 1e-3,
                      beta1= 0.9,
                      beta2= 0.999,
                      eps  = 1e-8) {
  state$t <- state$t + 1L
  state$m <- beta1 * state$m + (1 - beta1) * gw
  state$v <- beta2 * state$v + (1 - beta2) * (gw * gw)
  m_hat <- state$m / (1 - beta1^state$t)
  v_hat <- state$v / (1 - beta2^state$t)
  w_new  <- w - lr * m_hat / (sqrt(v_hat) + eps)
  list(w = w_new, state = state)
}

optimize_w_adam <- function(w_init, h, X, y, eta,
                            K    = 50,
                            lr   = 1e-2,
                            beta1= 0.9,
                            beta2= 0.999,
                            eps  = 1e-8) {
  w     <- w_init
  state <- init_adam(length(w))
  for (k in seq_len(K)) {
    gw    <- dE(w, h)
    upd   <- adam_step(w, gw, state, lr, beta1, beta2, eps)
    w     <- upd$w
    state <- upd$state
  }
  w
}

# ——— Main script ———

# reproducibility
rand_num1 <-9900; 
set.seed(rand_num1); 
print(rand_num1)
#source('perceptron_pruning_v5.1.r');
code_name<-'perceptron_pruning_v5.1.r';
#
N=500; # number of parameters 
p0 <-0.5; #sparsity
M    <- 10^3        # sample size
M_test    <- 10^3   # test sample size 
T    <- 100;        	# max. number of iterations
eta_set <- seq(0, 0.001, length.out = 11);
rho_set <- seq(0, 0.001, length.out = 11);
in_file<-paste(rand_num1,"_param.txt",sep="");
write(paste("r-code=", code_name,", rand_num1=",rand_num1, ", N=", N, ", p0=", p0,", M=", M, ", M_test=",M_test, ", T=",T, ", eta=[", eta_set[1],",...,",eta_set[length(eta_set)],"]", ", rho=[", rho_set[1],",...,",rho_set[length(rho_set)],"]", sep=""), in_file, ncolumns =1, append = FALSE,  sep = "\n");
out_file<-paste(rand_num1,"_stats.csv", sep="");
write(paste("eta","rho", "it","sum_h","Hamming", "MSE", "E", "train_error", "test_error",sep = "\t"), out_file, ncolumns =9, append = FALSE,  sep = "\t");

#prepare vectors and matrices
D_N <- diag(1, N, N);
zero <- rep(0, N);
#prepare vectors and matrices
D_M <- diag(1, M, M);
zero_M <- rep(0, M);
N1   <- floor(N*p0); #true number of parameters

L_train <- matrix(0, length(eta_set), length(rho_set)); #define matrix to store input
L_test <- matrix(0, length(eta_set), length(rho_set));
ROW=1;

#sample true parameters
w0   <- mvrnorm(1, zero, D_N); #sample parameters 
h0 <- sample(c(rep(1, N1), rep(0, N - N1)));#generate random binary vector with exactly N1 of 1's
#generate data for training
X    <- mvrnorm(M, zero, D_N)/sqrt(N); #sample inputs 

noise   <- mvrnorm(1, zero_M, D_M); #sample channel noise for linear regime comment o.w.
sigma0=0.01; #for linear regime set to 0 o.w.
y    <- phi(X %*% (w0 * h0))+sigma0*noise; #calculate outputs
#generate data for testing
X_test    <- mvrnorm(M_test, zero, D_N)/sqrt(N); 
y_test    <- phi(X_test %*% (w0 * h0));

start_time <- Sys.time(); 
print(start_time);
for (eta in eta_set)
{
	COL=1;
	for (rho in rho_set)
	{
	 
		# storage
		E_h    <- numeric(T);


		# compute initial w with no pruning
		w1 <- rnorm(N)
		w1 <- optimize_w_adam(w1, rep(1, N), X, y, eta, K=100, lr=1e-2)
		h1 <- rep(1, N); 

		E_diff=1.0;
		it=1;
		while ( E_diff>0 && it <= T) #
		{

			# —— coordinate search on h
			for (i in seq_len(N)) 
			{
				j <- sample.int(N, 1)
				h2 <- h1; 
				h2[j] <- 1 - h2[j]; #flip h[j]

				w2 <- optimize_w_adam(w1, h2, X, y, eta, K = 20, lr= 1e-2);

				delta <- E(w2, h2) - E(w1, h1) + 0.5 * rho * (h2[j] - h1[j]); #compute energy difference

				if (delta < 0) 
				{
					h1 <- h2
					w1 <- w2
				}
			}
			E_h[it] <- E(w1, h1) + 0.5 * rho * sum(h1);
			if (it>1)
			{
				E_diff=E_h[it-1]-E_h[it];
			}
			cat("it=", it, ", E(h|D)=", E_h[it],", ||h-h0||^2=", sum((h1-h0)^2), "\n");
			it <- it + 1;
		}
		cat("\n"); 
		train_error=L(w1*h1)/M;
		L_train[ROW,COL]=train_error;
		#save train data 
		X_temp    <- X; 
		y_temp    <- y;
		#compute test error
		X    <- X_test; 
		y    <- y_test;
		test_error=L(w1*h1)/M_test;
		#recover train data
		X    <- X_temp; 
		y    <- y_temp;
		L_test[ROW,COL]=test_error;
		write(paste(eta,rho, it-1, sum(h1), sum((h1-h0)^2)/N, sum((w1*h1 - w0*h0)^2)/N,E_h[it-1],train_error,test_error, sep = "\t"), out_file, ncolumns =1, append =TRUE,  sep = "\t");
		COL=COL+1;
	}
	ROW=ROW+1;
}
end_time <- Sys.time(); print(end_time - start_time);



