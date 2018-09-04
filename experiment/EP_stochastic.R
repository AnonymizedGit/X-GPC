# library(doMC)

# Global variable that indicates the current repetition

rep <- 1
t0 <- NULL

#eps <- NULL
#sign <- NULL

step_rate <- 1
decay <- 0.9
eps <- 1e-5
gms <- NULL
sms <- NULL
step <- NULL

##
# Function which computes the cholesky decomposition of the inverse
# of a particular matrix.
#
# @param	M	m x m positive definite matrix.
#
# @return	L	m x m upper triangular matrix such that
#			M^-1 = L %*% t(L)
#

cholInverse <- function(M) { rot180(forwardsolve(t(chol(rot180(M))), diag(nrow(M)))) }

##
# Function which rotates a matrix 180 degreees.
#

rot180 <- function(M) { matrix(rev(as.double(M)), nrow(M), ncol(M)) }

##
# This function computes the covariance matrix for the GP
#

kernel <- function(X, l, sigma0, sigma) {
	X <- X / matrix(sqrt(l), nrow(X), ncol(X), byrow = TRUE)
	distance <- as.matrix(dist(X))^2
	sigma * exp(-0.5 * distance) + diag(sigma0, nrow(X)) + diag(rep(1e-10, nrow(X)))
}

##
# Function which computes the kernel matrix between the observed data and the test data
#

kernel_nm <- function(X, Xnew, l, sigma) {

	X <- X / matrix(sqrt(l), nrow(X), ncol(X), byrow = TRUE)
	Xnew <- Xnew / matrix(sqrt(l), nrow(Xnew), ncol(Xnew), byrow = TRUE)
	n <- nrow(X)
	m <- nrow(Xnew)
	Q <- matrix(apply(X^2, 1, sum), n, m)
	Qbar <- matrix(apply(Xnew^2, 1, sum), n, m, byrow = T)
	distance <- Qbar + Q - 2 * X %*% t(Xnew)
	sigma * exp(-0.5 * distance)
}

##
# Function which computes the diagonal of the kernel matrix for the data
# points.
#
# @param	X 		n x d matrix with the n data points.
# @param	sigma		scalar with the amplitude of the GP.
# @param	sigma0		scalar with the noise level in the GP.
#
# @return	diagKnn		n-dimensional vector with the diagonal of the
#				kernel matrix for the data points.
#

computeDiagKernel <- function(X, sigma, sigma0) { rep(sigma, nrow(X)) + 1e-10 + sigma0 }

##
# Function that initializes the struture with the problem information.
#
# @param	X	n x d matrix with the data points.
# @param	Xbar	m x d matrix with the pseudo inputs.
# @param	sigma	scalar with the log-amplitude of the GP.
# @param	sigma0	scalar with the log-noise level in the GP.
# @param	l	d-dimensional vector with the log-lengthscales.
#
# @return	gFITCinfo	List with the problem information
#

initialize_kernel_FITC <- function(Y, X, Xbar, sigma, sigma0, l) {

	# We initialize the structure with the data and the kernel
	# hyper-parameters

	gFITCinfo <- list()
	gFITCinfo$X <- X
	gFITCinfo$Y <- Y
	gFITCinfo$Xbar <- Xbar
	gFITCinfo$m <- nrow(Xbar)
	gFITCinfo$d <- ncol(Xbar)
	gFITCinfo$n <- nrow(X)
	gFITCinfo$sigma <- sigma
	gFITCinfo$sigma0 <- sigma0
	gFITCinfo$l <- l

	# We compute the kernel matrices

	gFITCinfo$Kmm <- kernel(Xbar, gFITCinfo$l, gFITCinfo$sigma0, gFITCinfo$sigma)
	gFITCinfo$KmmInv <- chol2inv(chol(gFITCinfo$Kmm))
	gFITCinfo$Knm <- kernel_nm(X, Xbar, gFITCinfo$l, gFITCinfo$sigma)
	gFITCinfo$P <- gFITCinfo$Knm
	gFITCinfo$R <- cholInverse(gFITCinfo$Kmm)
	gFITCinfo$PRt <- gFITCinfo$P %*% t(gFITCinfo$R)
	gFITCinfo$PRtR <- gFITCinfo$PRt %*% gFITCinfo$R

	# We compute the diagonal matrices

	gFITCinfo$diagKnn <- computeDiagKernel(X, gFITCinfo$sigma, gFITCinfo$sigma0)
	gFITCinfo$diagPRtRPt <- gFITCinfo$PRt^2 %*% rep(1, gFITCinfo$m)
	gFITCinfo$D <- gFITCinfo$diagKnn - gFITCinfo$diagPRtRPt

	gFITCinfo
}


##
# Function that computes classes and probabilities of the labels of test data
#
# ret is the list returned by EP
#

predict <- function(Xtest, ret) {

	# We compute the FITC prediction

	posterior <- list(mNew = ret$mNew, vNew = ret$vNew)

	P_new <- kernel_nm(Xtest, ret$Xbar, ret$l, ret$sigma)
	diagKnn_new <- computeDiagKernel(Xtest, ret$sigma, ret$sigma0)

	PRtR_new <- P_new %*% ret$KmmInv

	gamma <- colSums(t(PRtR_new) * (posterior$vNew %*% t(PRtR_new)))

	z <- diagKnn_new - rowSums(PRtR_new * P_new) + gamma
	theta <- PRtR_new %*% posterior$mNew

	mPrediction <- theta
	vPrediction <- z + 1

	pnorm(mPrediction / sqrt(vPrediction))
}

##
# This function computes the gradients of the ML approximation provided by EP once it has converged
#

computeGradsPrior <- function(vNew, mNew, l, sigma0, sigma, gFITCinfo, indpointsopt) {

	Kinv <- gFITCinfo$KmmInv
	K <- gFITCinfo$Kmm
	M <- Kinv - Kinv %*% (vNew %*% Kinv) - ((Kinv %*% mNew) %*% (t(mNew) %*% Kinv))

	# We compute the derivatives of the kernel with respect to log_sigma

	dKmm_dlog_sigma0 <- diag(gFITCinfo$m) * sigma0

	gr_log_sigma0 <- -0.5 * sum(M * dKmm_dlog_sigma0)

	# We compute the derivatives of the kernel with respect to log_sigma0

	dKmm_dlog_sigma <- gFITCinfo$Kmm - diag(rep(gFITCinfo$sigma0, gFITCinfo$m)) - diag(1e-10, gFITCinfo$m)

	gr_log_sigma <- -0.5 * sum(M * dKmm_dlog_sigma)


	# The distance is v^2 1^T - 2 v v^T + 1^T v^2

	Ml <- 0.5 * M * gFITCinfo$Kmm
	Xl <-  (gFITCinfo$Xbar / matrix(sqrt(l), nrow(gFITCinfo$Xbar), ncol(gFITCinfo$Xbar), byrow = TRUE))
	gr_log_l <- - 2 * 0.5 * colSums(Ml %*% Xl^2) + 2 * 0.5 * colSums(Xl * (Ml %*% Xl))

	if (indpointsopt) {
		Xbar <- (gFITCinfo$Xbar / matrix(l, nrow(gFITCinfo$Xbar), ncol(gFITCinfo$Xbar), byrow = TRUE))
		Mbar <- t(M) * (- (gFITCinfo$Kmm - diag(gFITCinfo$sigma0, gFITCinfo$m) - diag(1e-10, gFITCinfo$m)))
		gr_xbar <- - 0.5 * 2 * (Xbar * matrix(rep(1, gFITCinfo$m) %*% Mbar, gFITCinfo$m, length(l)) - Mbar %*% Xbar)
	}
	else
	{gr_xbar <- 0}
	list(gr_log_l = gr_log_l, gr_log_sigma0 = gr_log_sigma0, gr_log_sigma = gr_log_sigma, gr_xbar = gr_xbar)
}

computeGradsLikelihood <- function(vNew, mNew, gFITCinfo, l, sigma0, sigma, eta1, eta2,indpointsopt) {

	# Loop through the data

	log_evidence <- 0
	n <- gFITCinfo$n
	m <- gFITCinfo$m

	# We precompute some derivatives

	dKnn_dlog_sigma0 <- rep(sigma0, gFITCinfo$n)
	dKmm_dlog_sigma0 <- diag(gFITCinfo$m) * sigma0
	dP_dlog_sigma0 <- matrix(0, gFITCinfo$n, gFITCinfo$m)

	dKnn_dlog_sigma <- rep(sigma, gFITCinfo$n)
	dKmm_dlog_sigma <- gFITCinfo$Kmm - diag(rep(gFITCinfo$sigma0, gFITCinfo$m)) - diag(1e-10, gFITCinfo$m)
	dP_dlog_sigma <- gFITCinfo$P

	# We compute dlog_evidece_dPRtR and dlog_evidece_dP and dlog_evidece_dKnn

	dlog_evidece_dPRtR <- matrix(0, n, m)
	dlog_evidece_dP <- matrix(0, n, m)
	dlog_evidece_dKnn <- rep(0, n)

	PRtRvNew <- gFITCinfo$PRtR %*% vNew
	PRtRvNewRtRPt <- rowSums(PRtRvNew * gFITCinfo$PRtR)
	C1 <- (eta2^-1 - PRtRvNewRtRPt)^-1
	PRtRvOldRtRPt <- PRtRvNewRtRPt + PRtRvNewRtRPt^2 * C1
	PRtRvOld <- PRtRvNew + matrix(PRtRvNewRtRPt * C1, n, m) * PRtRvNew
	PRtRmNew <- gFITCinfo$PRtR %*% mNew
	C2 <- eta2 * PRtRmNew - eta1
	PRtRmOld <- PRtRmNew + PRtRvOldRtRPt * C2
	mOldMatrix <- matrix(mNew, n, m, byrow = TRUE) + (PRtRvNew + PRtRvNew * matrix(PRtRvNewRtRPt * C1, n, m)) * matrix(C2, n, m)

	z <- gFITCinfo$D + PRtRvOldRtRPt + 1
	theta <- PRtRmOld
	logZ <-  pnorm(gFITCinfo$Y * theta / sqrt(z), log.p = TRUE)
	alpha <-  gFITCinfo$Y / sqrt(z) * exp(dnorm(gFITCinfo$Y * theta / sqrt(z), 0, 1, log = TRUE) -
		pnorm(gFITCinfo$Y * theta / sqrt(z), log.p = TRUE))

	dlog_evidece_dPRtR <- matrix(exp(-logZ + dnorm(gFITCinfo$Y * theta / sqrt(z), log = TRUE)) * gFITCinfo$Y, n, m) *
		(mOldMatrix / matrix(sqrt(z), n, m) - matrix(0.5 * theta * 1 / z^(3/2), n, m) * (-gFITCinfo$P + 2 * PRtRvOld))

	dlog_evidece_dP <- matrix(exp(-logZ + dnorm(gFITCinfo$Y * theta / sqrt(z), log = TRUE)) * gFITCinfo$Y, n, m) *
		(matrix(0.5 * theta * 1 / z^(3 / 2), n, m) * gFITCinfo$PRtR)
	dlog_evidece_dKnn <- exp(-logZ + dnorm(gFITCinfo$Y * theta / sqrt(z), log = TRUE)) *
		gFITCinfo$Y * (- 0.5 * theta * 1 / z^(3/2))

	# We now compute the actual gradients

	M1 <- gFITCinfo$KmmInv %*% t(dlog_evidece_dPRtR)
	M2 <- - M1 %*% gFITCinfo$PRtR
	M3 <- dlog_evidece_dP

	gr_log_sigma <- sum(t(M1) * dP_dlog_sigma) + sum(t(M2) * dKmm_dlog_sigma) + sum(t(dlog_evidece_dKnn) * dKnn_dlog_sigma) +
		sum(M3 * dP_dlog_sigma)

	gr_log_sigma0 <- sum(t(M1) * dP_dlog_sigma0) + sum(t(M2) * dKmm_dlog_sigma0) + sum(t(dlog_evidece_dKnn) * dKnn_dlog_sigma0) +
		sum(M3 * dP_dlog_sigma0)

	Ml <- 0.5 * (t(M1) + M3) * gFITCinfo$P
	Xbarl <-  (gFITCinfo$Xbar / matrix(sqrt(l), nrow(gFITCinfo$Xbar), ncol(gFITCinfo$Xbar), byrow = TRUE))
	Xl <-  (gFITCinfo$X / matrix(sqrt(l), nrow(gFITCinfo$X), ncol(gFITCinfo$X), byrow = TRUE))
	Ml2 <- t(M2) * gFITCinfo$Kmm * 0.5
	gr_log_l <- colSums(t(Ml) %*% Xl^2) - 2  * colSums(Xl * (Ml %*% Xbarl)) +  colSums(Ml %*% Xbarl^2) +
		colSums(t(Ml2) %*% Xbarl^2) - 2 * colSums(Xbarl * (Ml2 %*% Xbarl)) + colSums(Ml2 %*% Xbarl^2)

	if (indpointsopt) {
		Xbar <- (gFITCinfo$Xbar / matrix(l, nrow(gFITCinfo$Xbar), ncol(gFITCinfo$Xbar), byrow = TRUE))
		X <- (gFITCinfo$X / matrix(l, nrow(gFITCinfo$X), ncol(gFITCinfo$X), byrow = TRUE))
		Mbar <- t(M2) * - (gFITCinfo$Kmm - diag(gFITCinfo$sigma0, gFITCinfo$m) - diag(1e-10, gFITCinfo$m))
		Mbar2 <- (t(M1) + M3) * gFITCinfo$P
		gr_xbar <- (Xbar * matrix(rep(1, gFITCinfo$m) %*% Mbar, gFITCinfo$m, length(l)) - t(Mbar) %*% Xbar) +
				(Xbar * matrix(rep(1, gFITCinfo$m) %*% t(Mbar), gFITCinfo$m, length(l)) - Mbar %*% Xbar) +
				(t(Mbar2) %*% X) - ((Xbar * matrix(rep(1, gFITCinfo$n) %*% Mbar2, gFITCinfo$m, length(l))))
	}
	else{
		gr_xbar <- 0
	}
	list(gr_log_l = gr_log_l, gr_log_sigma0 = gr_log_sigma0, gr_log_sigma = gr_log_sigma, gr_xbar = gr_xbar)
}

##
# This function runs EP until convergence
#

epGPCInternal <- function(X, Y, inducingpoints, n_pseudo_inputs, Xtest, Ytest, minibatchsize, maxiter, indpointsopt, hyperparamopt, callback = 0) {

  t0 <<- proc.time()
	value_log <- data.frame(Iter=numeric(),Time=numeric(),Accuracy=numeric(),MeanLL=numeric(),MedianLL=numeric(),ELBO=numeric())
	nBatches <- nrow(X) / n_pseudo_inputs
  log_sigma <- 0
  log_sigma0 <- log(1e-3)
	#Xbar <- X[ sample(1 : nrow(X), n_pseudo_inputs), , drop = F ]
	Xbar <- inducingpoints
  log_l <- rep(log(estimateL(Xbar)), ncol(Xbar))

	l <- exp(log_l)
	sigma0 <- exp(log_sigma0)
	sigma <- exp(log_sigma)

	m <- nrow(Xbar)
	n <- nrow(X)

	# We split the data for each node

	eta1 <- rep(1, n)
	eta2 <- rep(1, n)
	eta_vectors <- matrix(0, m, n)

	vNew <- kernel(Xbar, exp(log_l), exp(log_sigma0), exp(log_sigma))
	KmmInv <- chol2inv(chol(vNew))
	mNew <- rep(0, m)

	gms <<- list(sigma = 0, sigma0 = 0, l = rep(0, length(log_l)), xbar = matrix(0, nrow(Xbar), ncol(Xbar)))
	sms <<- list(sigma = 0, sigma0 = 0, l = rep(0, length(log_l)), xbar = matrix(0, nrow(Xbar), ncol(Xbar)))
	step <<- list(sigma = 0, sigma0 = 0, l = rep(0, length(log_l)), xbar = matrix(0, nrow(Xbar), ncol(Xbar)))

	# We check for an initial solution

	# Main loop of EP
	damping <- 1
	convergence <- FALSE
	cont <- 1
	log_cont <- 1
	while (! convergence && cont <=  maxiter) {

			cat(".")

			#to_sel <- (round((n - 1) * nrow(X) / nBatches) + 1) : round(n * nrow(X) / nBatches)
			to_sel <- sample(x=n,size=minibatchsize,replace=FALSE)
			Xbatch <- X[ to_sel,, drop = FALSE ]
			Ybatch <- Y[ to_sel ]
			x_old <- eta_vectors[ , to_sel, drop = FALSE ]

			gFITCinfo <- initialize_kernel_FITC(Ybatch, Xbatch, Xbar, sigma, sigma0, l)

			x_new <- t(gFITCinfo$PRtR)

			x_newTvNew <- t(x_new) %*% vNew
			x_oldTvNew <- t(x_old) %*% vNew
			x_newTvNewx_old <- colSums(t(x_newTvNew) * x_old)
			x_oldTvNewx_old <- colSums(t(x_oldTvNew) * x_old)
			x_newTvNewx_new <- colSums(t(x_newTvNew) * x_new)

			C1 <- (eta2[ to_sel ]^-1 - x_oldTvNewx_old)^-1

			x_newTvOldx_new <- x_newTvNewx_new + x_newTvNewx_old^2 * C1
			x_oldTvOldx_new <- x_newTvNewx_old + x_oldTvNewx_old * x_newTvNewx_old * C1
			x_oldTmNew <- t(x_old) %*% mNew
			x_newTmNew <- t(x_new) %*% mNew
			C2 <- eta2[ to_sel ] * x_oldTmNew - eta1[ to_sel ]
			x_newTmOld <- x_newTmNew + x_oldTvOldx_new * C2

			z <- gFITCinfo$D + x_newTvOldx_new + 1
			theta <- x_newTmOld
			alpha <-  gFITCinfo$Y / sqrt(z) * exp(dnorm(gFITCinfo$Y * theta / sqrt(z), 0, 1, log = TRUE) -
				pnorm(gFITCinfo$Y * theta / sqrt(z), log.p = TRUE))

			eta2new <- (alpha^2 + alpha * theta / z) * (1 - (alpha^2 + alpha * theta / z) * x_newTvOldx_new)^-1
			eta1new <- eta2new * theta + alpha + alpha * x_newTvOldx_new * eta2new

			eta1new <- (1 - damping) *  eta1[ to_sel ] + damping * eta1new
			eta2new <- (1 - damping) *  eta2[ to_sel ] + damping * eta2new

			# This avoids uniform approximate factors

			eta2new[ abs(eta2new) < 1e-10] <- 1e-10

			# We update the posterior approximation

			if (any(eta2[ to_sel ] != 0))
				tryCatch(
				vOld <- vNew + (t(x_oldTvNew) %*% solve(diag(eta2[ to_sel ]^-1) - x_oldTvNew %*% x_old)) %*% x_oldTvNew
				, error = function(x) browser())
			else
				vOld <- vNew

			x_oldTvOld <- t(x_old) %*% vOld
			mOld <- mNew + t(x_oldTvOld) %*% ((matrix(eta2[ to_sel], length(to_sel), m) * t(x_old)) %*% mNew - eta1[ to_sel ])

			x_newTvOld <- t(x_new) %*% vOld
			tryCatch(
			vNew <- vOld - t(x_newTvOld) %*% solve(diag(c(eta2new)^-1) + x_newTvOld %*% x_new) %*% x_newTvOld
			, error = function(x) browser())
			x_newTvNew <- t(x_new) %*% vNew
			mNew <- mOld - t(x_newTvNew) %*% ((matrix(eta2new, length(to_sel), m) * t(x_new)) %*% mOld - eta1new)

			eta2[ to_sel ] <- eta2new
			eta1[ to_sel ] <- eta1new

			# We obtain the gradients
			if (hyperparamopt) {
				grad_prior <- computeGradsPrior(vNew, mNew, l, sigma0, sigma, gFITCinfo,indpointsopt)
				grad_likelihood <- computeGradsLikelihood(vNew, mNew, gFITCinfo, l, sigma0, sigma, eta1[ to_sel ], eta2[ to_sel ],indpointsopt)

				grad <- grad_prior
				grad$gr_log_l <- grad$gr_log_l + nBatches * grad_likelihood$gr_log_l
				grad$gr_log_sigma <- grad$gr_log_sigma + nBatches * grad_likelihood$gr_log_sigma
				grad$gr_log_sigma0 <- grad$gr_log_sigma0 + nBatches * grad_likelihood$gr_log_sigma0
				if (indpointsopt){
					grad$gr_xbar <- grad$gr_xbar + nBatches * grad_likelihood$gr_xbar
				}
				gms$sigma0 <- decay * gms$sigma0 + (1 - decay) * grad$gr_log_sigma0^2
				gms$sigma <- decay * gms$sigma + (1 - decay) * grad$gr_log_sigma^2
				gms$l <- decay * gms$l + (1 - decay) * grad$gr_log_l^2
				if (indpointsopt) {
					gms$xbar <- decay * gms$xbar + (1 - decay) * grad$gr_xbar^2
				}
				step$sigma0 <- sqrt(sms$sigma0 + eps) / sqrt(gms$sigma0 + eps) * grad$gr_log_sigma0 * step_rate
				step$sigma <- sqrt(sms$sigma + eps) / sqrt(gms$sigma + eps) * grad$gr_log_sigma * step_rate
				step$l <- sqrt(sms$l + eps) / sqrt(gms$l + eps) * grad$gr_log_l * step_rate
				if (indpointsopt){
					step$xbar <- sqrt(sms$xbar + eps) / sqrt(gms$xbar + eps) * grad$gr_xbar * step_rate
				}
				l <- exp(log(l) + step$l)
				sigma0 <- exp(log(sigma0) + step$sigma0)
				sigma <- exp(log(sigma) + step$sigma)
				if (indpointsopt){
					Xbar <- Xbar + step$xbar
				}
				sms$sigma0 <- decay * sms$sigma0 + (1 - decay) * step$sigma0^2
				sms$sigma <- decay * sms$sigma + (1 - decay) * step$sigma^2
				sms$l <- decay * sms$l + (1 - decay) * step$l^2
				if (indpointsopt){
					sms$xbar <- decay * sms$xbar + (1 - decay) * step$xbar^2
				}
				# We update the posterior distribution

				KmmNew <- kernel(Xbar, l, sigma0, sigma)
				KmmInvNew <- chol2inv(chol(KmmNew))
				vNewInv <- chol2inv(chol(vNew))
				vNew <-  chol2inv(chol(vNewInv - KmmInv + KmmInvNew))
				KmmInv <- KmmInvNew
				mNew<- vNew %*% (vNewInv %*% mNew)
			}
			eta_vectors[ ,to_sel ] <- x_new

			cont <- cont + 1

			if (is.function(callback) && (cont %% max(1,10^(floor(log10(cont/10)))) == 0)) {
				cat("\tIteration : ",  cont, "\n")
				value_log[nrow(value_log)+1,] <- callback(Xtest,Ytest,cont,t_0,computeEvidence(X, Y, Xbar, sigma, sigma0, l, vNew, mNew, eta1, eta2, eta_vectors, KmmInv),list(l = l, sigma0 = sigma0, sigma = sigma, mNew = mNew, vNew = vNew, KmmInv = KmmInv, Xbar = Xbar))
			}

	}
	if (is.function(callback)) {list(log_table=value_log[,],l = l, sigma0 = sigma0, sigma = sigma, X = X, Y = Y, mNew = mNew, vNew = vNew, KmmInv = KmmInv, Xbar = Xbar)}
	else {list(l = l, sigma0 = sigma0, sigma = sigma, X = X, Y = Y, mNew = mNew, vNew = vNew, KmmInv = KmmInv, Xbar = Xbar)}
}

##
# Function that estimates the initial lengthscale value

estimateL <- function (X) {

	D <- as.matrix(dist(X))
	median(D[ upper.tri(D) ])
}

# Function to save metrics during model training

save_log <- function(Xtest,Ytest,iter,t_0,evidence, model) {
	t_before <- proc.time()
	prediction <- predict(Xtest, model)
	t_after <- proc.time()

	t0 <<- t0 + (t_after - t_before)

	acc <- 1.0-mean(Ytest != sign(prediction - 0.5))
	# We evaluate the test log-likelihood

	meanll <- mean(log(prediction * (Ytest == 1) + (1 - prediction) * (Ytest == -1)))
	medianll <- median(log(prediction * (Ytest == 1) + (1 - prediction) * (Ytest == -1)))
	list(iter,proc.time()[1]-t0[1],acc,meanll,medianll,evidence)
	#write.table(t(c(error, sum(ll), proc.time() - t0)), file =
		#filename, row.names = F, col.names = F, append = TRUE)


}
###
# Function which computes the EP approximation of the log evidence.
#
# @param	f1Hat		The approximation for the first factor.
# @param	gFITCinfo	The list with the problem information.
# @param	Y		The class labels.
#
# @return	logZ		The log evidence.
#

computeEvidence <- function(X, Y, Xbar, sigma, sigma0, l, vNew, mNew, eta1, eta2, eta_vectors, KmmInv) {

	log_det_vNew <- 2 * sum(log(diag(chol(vNew))))
	Vinv <- chol2inv(chol(vNew))


	# Loop through the data

	log_evidence <- 0

	n <- nrow(X)
	m <- ncol(vNew)

	Xbatch <- X
	Ybatch <- Y
	x_old <- eta_vectors

	gFITCinfo <- initialize_kernel_FITC(Ybatch, Xbatch, Xbar, sigma, sigma0, l)

	x_new <- t(gFITCinfo$PRtR)

	x_newTvNew <- t(x_new) %*% vNew
	x_oldTvNew <- t(x_old) %*% vNew
	x_newTvNewx_old <- colSums(t(x_newTvNew) * x_old)
	x_oldTvNewx_old <- colSums(t(x_oldTvNew) * x_old)
	x_newTvNewx_new <- colSums(t(x_newTvNew) * x_new)

	C1 <- (eta2^-1 - x_oldTvNewx_old)^-1

	x_newTvOldx_new <- x_newTvNewx_new + x_newTvNewx_old^2 * C1
	x_oldTvOldx_new <- x_newTvNewx_old + x_oldTvNewx_old * x_newTvNewx_old * C1
	x_oldTvOldx_old <- x_oldTvNewx_old + x_oldTvNewx_old^2 * C1
	x_oldTmNew <- t(x_old) %*% mNew
	x_newTmNew <- t(x_new) %*% mNew
	x_oldTmNew <- t(x_old) %*% mNew
	C2 <- eta2 * x_oldTmNew - eta1
	x_newTmOld <- x_newTmNew + x_oldTvOldx_new * C2
	x_oldTmOld <- x_oldTmNew + x_oldTvOldx_old * C2

	mOldVinvmOld <- sum(t(mNew) %*% Vinv %*% mNew) + 2 * (x_oldTmNew + x_oldTmNew * x_oldTvNewx_old * C1) * C2 +
			(x_oldTvNewx_old+ x_oldTvNewx_old^2 * C1) * C2^2 + (x_oldTvNewx_old^2 * C1 + x_oldTvNewx_old^3 * C1^2) * C2^2

	z <- gFITCinfo$D + x_newTvOldx_new + 1
	theta <- x_newTmOld

	logZ <-  pnorm(Y * theta / sqrt(z), log.p = TRUE) + m / 2 * log(2 * pi) +
		0.5 * mOldVinvmOld - 0.5 * eta2 * x_oldTmOld^2 +
		0.5 * log_det_vNew - 0.5 * log(1 - eta2 * x_oldTvNewx_old)

	logZtilde <-  m / 2 * log(2 * pi) + 0.5 * sum(mNew * (Vinv %*% mNew)) + 0.5 * log_det_vNew

	log_evidence <- sum(logZ - logZtilde)

	log_evidence <- log_evidence + m / 2 * log(2 * pi) + 0.5 * sum(mNew * (Vinv %*% mNew)) + 0.5 * log_det_vNew
	log_evidence <- log_evidence - m / 2 * log(2 * pi) - 0.5 * sum(- 2 * log(diag(chol(KmmInv))))

	log_evidence
}
