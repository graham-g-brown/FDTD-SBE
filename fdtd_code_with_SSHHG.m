set(0,'DefaultFigureWindowStyle','docked')

au2eV = 27.211396;  % atomic units to eV
femtosecondAU  = 41.341374575751; % Femtosecond to atomic units (time)
 
t_global = tic;
fprintf('=== Simulation started at %s ===\n', datestr(now,'yyyy-mm-dd HH:MM:SS'));
 
print_mem_snapshot('start of script');
 
E_preset_flag      = false;  % kept for future use (not used in this pipeline)
is_normalized_flag = false;  % reserved

system = 'SBE' ;

if strcmp(system, 'SBE')

    [ H , P , k ] = SBE_bloch_basis_matrix_elements ( ) ;


    N_t = 8192
    w0 = 0.014225
    F0 = 0.003
    t = linspace ( - 0.5 , 0.5 , N_t ) * 16 * 2 * pi / w0

    F = F0 * exp ( - t .* t / (32 * 32) * w0 ) .* sin ( w0 * t )


    J_single = SBE_current_from_field ( F , t , H , P , k )

end

%---- CONSTANTS ----
c0      = 299792458;
lambda0 = 800e-9;
w0      = 2*pi*c0/lambda0;
 
% ---- GLOBAL SIMULATION CFG (single source of truth) ----
cfg = struct();
 
cfg.sim.Tprop_fs     = 8;         % total simulated time (fs)
cfg.grid.dx          = 1.8135e-9;  % m
cfg.grid.Courant     = 1/10;
 
cfg.src.lambda_nm    = 800;
cfg.src.lambda0      = lambda0;
cfg.src.w0           = w0;
 
% timebase
cfg.time.dt   = cfg.grid.Courant * cfg.grid.dx / c0;
cfg.time.Nt   = ceil((cfg.sim.Tprop_fs*1e-15)/cfg.time.dt);
cfg.time.t    = (0:cfg.time.Nt-1) * cfg.time.dt;  % seconds
 
% ---- PTC / INDEX VS TIME (UNIFORM) ----
ptc.use       = false;    % set true if you want n(t) modulation
ptc.n0        = 2.0;
ptc.dN        = 0.8;
ptc.Omega_THz = 375;
ptc.waveform  = 'cos';    % 'cos' | 'square' | 'gaussian_kick'
 
dt    = cfg.time.dt;
t_vec = cfg.time.t;
 
sig      = (25e-15)/sqrt(8*log(2));  % example 25 fs
t0_pulse = 6*sig;
 
switch lower(ptc.waveform)
    case 'cos'
        n_t = ptc.n0 + ptc.dN*cos(2*pi*ptc.Omega_THz*1e12*t_vec);
    case 'square'
        n_t = ptc.n0 + ptc.dN*sign(sin(2*pi*ptc.Omega_THz*1e12*t_vec));
    case 'gaussian_kick'
        Tm  = t0_pulse; sigm = sig/2;
        n_t = ptc.n0 + ptc.dN*exp(-0.5*((t_vec-Tm)/sigm).^2);
    otherwise
        n_t = ptc.n0 + 0*t_vec;
end
if ~ptc.use
    n_t = ptc.n0 + 0*t_vec;
end
epsr_t = n_t.^2;
 
%% ------------------------------------------------------------------------
% STEP 1: FDTD warmup (plane wave, 20 fs, no sources)
%         record E(x,t) downsampled + E(t) at every cell
%% ------------------------------------------------------------------------
 
fdtd_cfg = struct();
fdtd_cfg.dx           = cfg.grid.dx;
 
lambda_med            = lambda0/ptc.n0;
fdtd_cfg.Nx           = max(64, round(lambda_med/fdtd_cfg.dx));
fdtd_cfg.boundary     = 'periodic';
fdtd_cfg.store_stride = 4;   % temporal downsampling for E(x,t) plot
 
fdtd_cfg.probe_cell_frac   = 0.7;   % probe near the right
fdtd_cfg.sample_stride_x   = 1;     % record E(t) every cell
 
fdtd_cfg.ic = struct();
fdtd_cfg.ic.enable = true;          % initial forward plane wave
fdtd_cfg.ic.kind   = 'forward';
fdtd_cfg.ic.E0     = 3e8;           % amplitude (V/m)
 
% ==== Time-dependent gain parameters (we'll build sigma_t from these)
fdtd_cfg.gain_peak_sigma = -5.5e4;  % S/m  (negative = gain)
fdtd_cfg.gain_rise_fs    = 6;       % ramp up 0 -> sigma_peak over 6 fs
fdtd_cfg.gain_fall_fs    = 12;      % ramp back to 0 by 12 fs total
 
% FDTD warmup in step 1
t_fdtd1 = tic;
fdtd1_out = fdtd_only_periodic_gain(cfg, fdtd_cfg, epsr_t);
 
fprintf('Finished FDTD #1 in %.2f s (total %.2f s)\n', ...
        toc(t_fdtd1), toc(t_global));
print_mem_snapshot('after FDTD #1');
 
figure; 
subplot(1,2,1)
plot(fdtd1_out.t*1e15, fdtd1_out.Eprobe);
xlabel('t (fs)'); ylabel('E_{probe} (V/m)');
title('Step 1: probe field with time-dependent gain');
grid on;
 
subplot(1,2,2)
imagesc(fdtd1_out.t_store*1e15, fdtd1_out.x*1e6, fdtd1_out.E_xt);
axis xy;
xlabel('t (fs)');
ylabel('x (\mum)');
title('Step 1: E(x,t) (downsampled)');
colorbar; ylabel(colorbar,'E (V/m)');
 
%% ------------------------------------------------------------------------
% STEP 2: TDSE – build current J(t) for each sampled field for FDTD #2
%         Input: E_sample_mat(t,x), t(t)
%         Output: J_src_mat(t,x) for FDTD
%% ------------------------------------------------------------------------
 
E_sample_mat = fdtd1_out.E_sample_mat;   % [Nt x Ns]
sample_cells = fdtd1_out.sample_cells;   % [1 x Ns]
t_tdse       = cfg.time.t;               % same time grid
 
[Nt_TDSE, Ns] = size(E_sample_mat);
J_src_mat = zeros(Nt_TDSE, Ns);
 
tic_tdse = tic;
print_mem_snapshot('before TDSE block');
parfor s = 1:Ns
    E_local = E_sample_mat(:,s);
    
    if strcmp(system, 'SBE')

        J_src_mat(:,s) = SBE_current_from_field ( E_local , t_tdse , H , P , k ) ;
    else
        J_src_mat(:,s) = tdse_current_from_field(E_local, t_tdse);
    end
end
fprintf('Finished TDSE block in %.2f s (total %.2f s)\n', ...
        toc(tic_tdse), toc(t_global));
print_mem_snapshot('after TDSE block');
 
%% ------------------------------------------------------------------------
% STEP 3: drive second FDTD with TDSE current sources
%% ------------------------------------------------------------------------
 
t_fdtd2 = tic;
% Second FDTD: same grid, no IC, driven only by currents
fdtd_cfg2 = fdtd_cfg;
fdtd_cfg2.ic.enable = false;   % no initial plane wave
 
fdtd2_out = fdtd_only_periodic_current(cfg, fdtd_cfg2, epsr_t, J_src_mat, sample_cells);
 
fprintf('Finished FDTD #2 in %.2f s (total %.2f s)\n', ...
        toc(t_fdtd2), toc(t_global));
print_mem_snapshot('after FDTD #2');
 
fprintf('=== Simulation finished at %s (total %.2f s) ===\n', ...
        datestr(now,'yyyy-mm-dd HH:MM:SS'), toc(t_global));
 
%% ------------------------------------------------------------------------
% STEP 4: FDTD-based Plots
%% ------------------------------------------------------------------------
 
% 4A: E(x,t) downsampled from step 1
figure('Color','w','Name','Step 1: E(x,t) (warmup FDTD)');
imagesc(fdtd1_out.t_store*1e15, fdtd1_out.x*1e6, fdtd1_out.E_xt);
axis xy;
xlabel('t (fs)');
ylabel('x (\mum)');
title('Step 1: E(x,t) (downsampled)');
colorbar; ylabel(colorbar,'E (V/m)');
 
% 4D: E(x,t) from step 3 FDTD
figure('Color','w','Name','Step 3: E(x,t) from current-driven FDTD');
imagesc(fdtd2_out.t_store*1e15, fdtd2_out.x*1e6, fdtd2_out.E_xt);
axis xy;
xlabel('t (fs)');
ylabel('x (\mum)');
title('Step 3: E(x,t) (current-driven FDTD)');
colorbar; ylabel(colorbar,'E (V/m)');
 
%% ------------------------------------------------------------------------
% STEP 5: Spatial and temporal spectra from step-3 FDTD
%% ------------------------------------------------------------------------
 
% Spatial FFT at t=0 and t=end (approx)
E_start = fdtd2_out.E_xt(:,1);
E_end   = fdtd2_out.E_xt(:,end);
x       = fdtd2_out.x;
dx      = fdtd2_out.dx;
L       = numel(x)*dx;
Nx_sp   = numel(x);
dk      = 2*pi/L;
% integer mode indices consistent with fftshift
if mod(Nx_sp,2) == 0
    m_idx = -Nx_sp/2 : Nx_sp/2-1;          % even N
else
    m_idx = -(Nx_sp-1)/2 : (Nx_sp-1)/2;    % odd N
end
k      = m_idx * dk;
 
E0k   = fftshift(fft(ifftshift(E_start)))/Nx_sp;
Eendk = fftshift(fft(ifftshift(E_end)))/Nx_sp;
S0k   = abs(E0k).^2; S0k = S0k./max(S0k+eps);
Sendk = abs(Eendk).^2; Sendk = Sendk./max(Sendk+eps);
 
% express as spatial harmonic order m = k/k0_med
n_med  = sqrt(epsr_t(1));
 
% numerical cavity fundamental
k_cav   = 2*pi / L;     % exactly the mode used in the initial condition
m_axis  = k / k_cav;
 
figure('Color','w','Name','Spatial spectra of E(x) at start & end (step 3)');
subplot(1,2,1);
semilogy(m_axis, S0k + 1e-30, 'LineWidth', 1.2);
xlabel('m = k / k_0');
ylabel('Norm. |E_k|^2');
title('Start (t \approx 0)');
grid on;
xlim([0 100])
 
subplot(1,2,2);
semilogy(m_axis, Sendk + 1e-30, 'LineWidth', 1.2);
xlabel('m = k / k_0');
ylabel('Norm. |E_k|^2');
title('End (t \approx T)');
grid on;
xlim([0 100])
 
% Temporal FFT of probe location (step 3)
Eprobe = fdtd2_out.Eprobe(:).';
t_fdtd = fdtd2_out.t(:).';
[Sp, qp] = simple_hhg_spectrum(Eprobe, t_fdtd, lambda0);
 
figure('Color','w','Name','Temporal spectrum at probe (step 3)');
semilogy(qp, Sp + 1e-30, 'LineWidth', 1.4);
xlabel('Harmonic order q');
ylabel('Norm. power');
title('Temporal spectrum at probe (current-driven FDTD)');
xlim([0, 120]);
grid on;

%% ===================================================================== %%
%                            LOCAL FUNCTIONS                              %
%% ===================================================================== %%

function J_single = SBE_current_from_field ( E , t , H , P , k )

    N_k      = 64; 
    T2       = 1000;
    N_b      = 2 ;
    N_t      = size(t , 2)  ;
    dt       = t(2) - t(1)  ;
    dk       = k(2) - k(1)  ;
    J_single = zeros(N_t,1) ;

    rho_t = zeros(N_t, N_k, N_b, N_b);

    rho_t (1, :, 1, 1) = 1 / N_k ;

    A = - cumtrapz(t, E)  ; %getAfromE ( E , t ) ;

    plot ( t , E )
    pause 
    plot ( t , A )
    pause

    % ---------- Time propagation (RK4, length gauge) ----------
    for tdx = 2:N_t
        
        
        rho  = squeeze(rho_t(tdx-1,:,:,:));      % (N_k x N_b x N_b)
    
        % Stage fields
        A1 = A (tdx );

        if tdx < N_t
        
            A2 = A ( tdx - 1 ) + 0.5 * dt * 0.5 * ( A ( tdx - 1 ) + A ( tdx ) ) ;
            A3 = 0.5 * ( A ( tdx - 1 ) + A ( tdx ) );
            A4 = A(tdx);
        
        else

            A2 = 0;
            A3 = 0;
            A4 = 0;
        
        end
    
        % RK4 stages (length gauge)
        
        k1 = EOM_Velocity(rho                 , H , P , A1, N_b , T2 ) ;
        k2 = EOM_Velocity(rho + 0.5 * dt * k1 , H , P , A2, N_b , T2 ) ;
        k3 = EOM_Velocity(rho + 0.5 * dt * k2 , H , P , A3, N_b , T2 ) ;
        k4 = EOM_Velocity(rho +       dt * k3 , H , P , A4, N_b , T2 ) ;
    
        rho_next         = rho + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
        rho_t(tdx,:,:,:) = rho_next;
    
        % Current j(t) = sum_k Tr[ rho^k P^k ]  (gauge-invariant observable)
        j_temp = 0;
        for m = 1:N_b
            for n = 1:N_b
                j_temp = j_temp + sum( rho_next(:,m,n) .* squeeze(P(:,n,m)) );
            end
        end
        J_single(tdx) = real(j_temp);   % imaginary part should be ~0 numerically
    end

    
    figure
    plot ( J_single )
    pause

end

function drhodt = EOM_Velocity (rho, H, P, A, N_b, T2)
% Velocity-gauge SBE right-hand side (per time step)
% rho : (N_k x N_b x N_b) density matrix vs k
% H   : (N_k x N_b x N_b) Hamiltonian (typically diagonal in band index)
% P   : (N_k x N_b x N_b) momentum matrix elements
% A   : scalar vector potential A(t) at this time
% N_b : number of bands
% T2  : dephasing time (off-diagonal decay)
% drhodt - d(\rho)/dt
    drhodt = complex(zeros(size(rho)));

    for m = 1:N_b
        for n = 1:N_b
            % Coherent energy term: -i (h_mm - h_nn) rho_mn
            drhodt(:,m,n) = drhodt(:,m,n) ...
                + (-1i) * ( squeeze(H(:,m,m)) - squeeze(H(:,n,n)) ) .* squeeze(rho(:,m,n));

            % Field-driven commutator: -i A(t) [P, rho]_mn
            acc = complex(zeros(size(rho,1),1));
            for b = 1:N_b
                acc = acc ...
                    + squeeze(P(:,m,b)) .* squeeze(rho(:,b,n)) ...
                    - squeeze(rho(:,m,b)) .* squeeze(P(:,b,n));
            end
            drhodt(:,m,n) = drhodt(:,m,n) + (-1i) * A .* acc;

            % Phenomenological dephasing on off-diagonals
            if m ~= n
                drhodt(:,m,n) = drhodt(:,m,n) - squeeze(rho(:,m,n)) / T2;
            end
        end
    end
end


function [ H_k , P_k , k ] = SBE_bloch_basis_matrix_elements ( ) 

    % Number of points in spatial grid within unit cell
    N_x  = 128; %im
    % portant to choose enough for convergence.
    % Number of considered points in crystal momentum grid
    N_k  = 64;  % num. of k points determines the size of the lattice. you need enough lattice sites for the electron to not reach the edge of the lattice from the EM field dynamics.
    % Number of bands
    N_b  = 4; % the more bands, the heaviear the computation is later on in the HHG calculations.
    
    % Lattice constant
    a0   = 8.0;
    % Reciprocal Lattice vector
    b0   = 2.0 * pi / a0;
    % Strength of potential
    v0   = - 0.37;
    % Define spatial grid within single unit cell spanning [- a0 / 2, a0 / 2)
    x    = linspace(- a0 / 2.0, a0 / 2.0 - (a0/N_x), N_x);
    dx   = x(2) - x(1);
    
    % Define crystal momentum grid spanning single Brillouin Zone [- π / a0 , π / a0)
    k    = linspace(- pi / a0 , pi / a0 , N_k + 1 );
    dk   = k(2) - k(1);
    k   = k + 0.5 * dk;
    [X, K] = meshgrid(x, k);
    
    % Define conjugate reciprocal space grid G for spectral decomposition of potential and Bloch states
    G = 2.0 * pi * ((-N_x/2):(N_x/2 - 1)) / (N_x * dx);

    % Define matrix representing the potential in the conjugate domain
    V   = v0*diag(ones(N_x,1)) + v0*diag(ones(N_x-1,1),1)/2 + v0*diag(ones(N_x-1,1),-1)/2;
    v   = v0*(1+cos(2*pi / a0*x)); % potential in real space
    
    % figure(1) % displaying the potential in real space
    % plot(x/a0, v)
    
    %%
    % Define array to contain the energies for each band at each crystal momentum
    eps  = zeros( N_k + 1 , N_b);  % why this +1?
    % Define array to contain the representation of the Bloch states in the conjugate domain
    psi = zeros(N_k + 1 , N_b , N_x);
    
    % Loop over the crystal momentum and calculate the eigenstates and energies for each
    for kdx = 1:N_k + 1
        % Define the effective Hamiltonian for crystal momentum k to be used in the central equation
        H = 0.5 * diag((k(kdx) + G).^2) + V; % the Hamiltonian for crystal momentum k(kdx)
    
        % Solve central equation
        [eigenVectors, eigenValues] = eig(H);
    
        % Sort eigenvectors and eigenvalues in increasing order
        [sortedEigenValues, idx] = sort(diag(eigenValues));
        eigenValues = sortedEigenValues;
        eigenVectors = eigenVectors(:, idx);
    
        % % Save energies and eigenstates
        eps(kdx, :) = eigenValues(2:N_b+1); % its a common choice for this potential to take solutions from the second band and not from first band. 
        psi(kdx,:, :) = eigenVectors(:, 2:N_b+1).';
        
    end
    
    % we shift the energies such that  ϵ=0  occurs at the midpoint between the valence and conduction bands.
    energyShift = eps(floor(N_k/2) + 1, 2) - 0.5*(eps(floor(N_k/2)+1, 2) - eps(floor(N_k/2)+1, 1));
    eps = eps - energyShift;

    

    %%
    % Array to contain the real-space representation of the Bloch functions
    umk = zeros(size(psi));
    % For each crystal momentum and each band, calculate the real-space Bloch function
    for kdx = 1:(N_k + 1)
        for bdx = 1:N_b
            % Apply FFT with fftshift before and after 
            umk(kdx, bdx, :) = fftshift(fft(fftshift(psi(kdx, bdx, :))));
    
            % Normalize
            normFactor = sqrt(sum(abs(umk(kdx, bdx, :)).^2) * dx / a0);
            umk(kdx, bdx, :) = umk(kdx, bdx, :) / normFactor;
        end
    end

    for bdx = 1:N_b
        
        for kdx = 2:N_k+1
            % Compute phase difference between neighboring k-points
            phase = angle( sum( conj(umk(kdx, bdx, :)) .* umk(kdx - 1, bdx, :) ) * dx );
    
            % Apply phase correction
            umk(kdx, bdx, :) = umk(kdx, bdx, :) * exp(1i * phase);
        end

        % Handle periodic gauge closure: k = 1 vs k = N_k+1 (wraparound)
        kdx = 1;
        phase = angle( sum( umk(kdx, bdx, :) .* conj(umk(end, bdx, :)) ) * dx );
        umk(kdx, bdx, :) = umk(kdx, bdx, :) * exp(1i * phase);
    
        % Renormalize
        normFactor = sqrt( sum( abs(umk(kdx, bdx, :)).^2 ) * dx );
        umk(kdx, bdx, :) = umk(kdx, bdx, :) / normFactor;
    end

    %% second phase continuity - global continuity
    for bdx = 1:N_b  % Loop over bands
    
        % Get the wavefunctions at first and last k-points
        u_first = squeeze(umk(1, bdx, :));    % equivalent to umk[0, bdx, :]
        u_last  = squeeze(umk(end, bdx, :));  % equivalent to umk[-1, bdx, :]
    
        % Compute phase correction
        phase = -angle( sum( u_last .* exp(-1i * b0 * x(:)) .* conj(u_first) ) * dx );
    
        disp(phase);  % Output the phase value
    
        % Apply phase smoothing across all k-points
        for kdx = 1:N_k
            delta_k = k(kdx) - k(1);  % k[kdx] - k[0]
            umk(kdx, bdx, :) = squeeze(umk(kdx, bdx, :)) * exp(-1i * delta_k / b0 * phase);
            umk(kdx, bdx, :) = umk(kdx, bdx, :) / sqrt(sum(abs(umk(kdx, bdx, :)).^2) * dx / a0); %normalize - gpt additon
    
        end
    end
    
    % normalizing umk
    for kdx = 1:N_k
        for bdx = 1:N_b
            norm_factor = sqrt(sum(abs(umk(kdx, bdx, :)).^2) * dx/ a0);
            umk(kdx, bdx, :) = umk(kdx, bdx, :) / norm_factor;
        end
    end

    %%  Calculating matrix elements for time propagation  
    
    DFT = dftmtx(N_x);
    DFT_inv = inv(DFT);  % MATLAB warning says to consider using diag(k_x)/DFT instead of inv(DFT)*diag(k_x)
    
    % Frequency vector (same as np.fft.fftfreq in Python)
    k_x = 2.0 * pi * [0:(N_x/2 - 1), -N_x/2:-1] / (N_x * dx);
        
    D1_x = DFT_inv * ( - 1i * diag(k_x) ) * DFT / N_x / dx;

    
    % Allocate matrices
    P_k = zeros(N_k, 2 , 2 );
    H_k = zeros(N_k, 2 , 2 );
    
    for kdx = 1:N_k
        for bdx = 1:2
            for bbdx = 1:2

                P_k(kdx,bdx,bbdx) = - 1i * sum( conj( squeeze(umk(kdx, bbdx, :)) ) .* (D1_x * squeeze(umk(kdx, bdx, :))) ) * dx;
                
                if bdx == bbdx
                
                    bdx , bbdx
                    H_k(kdx, bdx, bbdx) = eps(kdx, bdx );
                    P_k(kdx, bdx, bbdx) = P_k(kdx, bdx , bbdx ) + k(kdx);
                
                end                
            end
        end
    end    
    k = k ( 1 : N_k )
end
 
%% ===================================================================== %%
%                            LOCAL FUNCTIONS                              %
%% ===================================================================== %%
 
function J_single = tdse_current_from_field(E_local, t_vec)
    % TDSE_CURRENT_FROM_FIELD
    % Given a single local electric field E_local(t) [V/m] and time axis
    % t_vec [s], compute the microscopic current density J_single(t)
    % [A/m^2] suitable as a source for FDTD.
 
    E_local = E_local(:);
    t_vec   = t_vec(:);
    if numel(t_vec) ~= numel(E_local)
        error('tdse_current_from_field: E_local and t_vec must have the same length.');
    end
 
    Nt = numel(t_vec);
    dt = mean(diff(t_vec));
 
    % atomic units (local to TDSE)
    au.E  = 5.14220674763e11;    % V/m
    au.t  = 2.4188843265857e-17; % s
    au.a0 = 5.29177210903e-11;   % m
    au.e  = 1.602176634e-19;     % C
 
    % TDSE grid and options (local to TDSE)
    increase        = 3;
    tdse.Nx        = increase*2^12;
    tdse.xmax_au   = increase*300;
    tdse.soft_a_au = 0.9;
    tdse.use_itp   = true;
 
    % static TDSE setup + ground state
    [tdse_static, psi0, Ip_au] = tdse_setup_static(tdse, au, dt, Nt); %#ok<NASGU>
 
    % propagate and get <x>(t)
    [~, x_expect_au] = tdse_propagate_from_ground(tdse_static, psi0, au, Nt, E_local);
 
    % gas density (local to TDSE block)
    N_m3 = 5e25;    % example gas density (m^-3) – adjust as needed
 
    % build current density from <x>(t)
    J_single_mat = build_J_from_x(x_expect_au, dt, au, N_m3);
    J_single = J_single_mat(:,1);    % [Nt x 1]
end
 
function [tdse_static, psi0, Ip_au] = tdse_setup_static(tdse, au, dt, Nt)
    % Build static TDSE structures and ground state once (shared for this
    % call).
    Nx    = tdse.Nx;
    xmax  = tdse.xmax_au;
    x_au  = linspace(-xmax, xmax, Nx).';
    dx_au = x_au(2)-x_au(1);
    dk_au = 2*pi/(Nx*dx_au);
    k_au  = fftshift((-Nx/2:Nx/2-1).' * dk_au);
    soft_a = tdse.soft_a_au;
 
    Vx     = -1./sqrt(x_au.^2 + soft_a^2);
    Kprop  = @(h) exp(-1i*0.5*(k_au.^2)*h);
 
    if tdse.use_itp
        [psi0, E0_au] = tdse_groundstate_itp(x_au, soft_a, ...
            'mask_edge', 0.05, 'dtau', 0.05, 'max_iter', 4000, ...
            'tol', 1e-10, 'report', true);
        fprintf('ITP: ground state E0 = %.6f a.u. (Ip ≈ %.2f eV)\n', ...
            E0_au, max(0,-E0_au)*27.211386);
        Ip_au = max(0, -E0_au);
    else
        psi0 = (1/pi)^(1/4)*exp(-x_au.^2/2);
        psi0 = psi0 / sqrt(trapz(x_au, abs(psi0).^2));
        Ip_au = 0;
    end
 
    dt_au = dt/au.t;
    dVdx  = x_au ./ ((x_au.^2 + soft_a^2).^(3/2));
 
    tdse_static = struct();
    tdse_static.x_au   = x_au;
    tdse_static.Vx     = Vx;
    tdse_static.Kprop  = Kprop;
    tdse_static.dVdx   = dVdx;
    tdse_static.soft_a = soft_a;
    tdse_static.dt_au  = dt_au;
    tdse_static.Nt     = Nt;
end
 
function [a_t, x_expect_au] = tdse_propagate_from_ground(tdse_static, psi0, au, Nt, E_mod)
    % Propagate TDSE from a given initial state psi0 for a given E_mod(t).
    psi  = psi0;
    E_au = E_mod(:).' / au.E;
 
    x_au   = tdse_static.x_au;
    Vx     = tdse_static.Vx;
    Kprop  = tdse_static.Kprop;
    dVdx   = tdse_static.dVdx;
    dt_au  = tdse_static.dt_au;
 
    a_t         = zeros(1,Nt);
    x_expect_au = zeros(1,Nt);
 
    for n = 1:Nt
        Veff = Vx + x_au*E_au(n);
        psi  = exp(-1i*Veff*dt_au/2).*psi;
        psi  = ifft( fft(psi).*Kprop(dt_au) );
        psi  = exp(-1i*Veff*dt_au/2).*psi;
        psi  = apply_mask(psi, 0.05);
        psi  = psi / sqrt(trapz(x_au, abs(psi).^2));
 
        a_t(n)         = -real(trapz(x_au, conj(psi).*dVdx.*psi)) - E_au(n);
        x_expect_au(n) = real(trapz(x_au, conj(psi).*x_au.*psi));
    end
end
 
function J_src_mat = build_J_from_x(x_au_vec, dt, au, N_m3)
    % Build current density J(t) from x(t) (a.u.) for a single trajectory.
    % x_au_vec: [Nt x 1] <x>(t) in a.u.
    % J_src_mat: [Nt x 1] current density A/m^2.
    x_au_vec = x_au_vec(:);
    Nt = numel(x_au_vec);
 
    J_src_mat = zeros(Nt, 1);
    x_SI   = x_au_vec * au.a0;        % m
    p_t    = -au.e * x_SI;            % dipole moment (C·m)
    J_single = gradient(p_t, dt);     % A·m
    J_src_mat(:,1) = N_m3 * J_single; % A/m^2
end
 
function [S_norm, q_axis] = simple_hhg_spectrum(a_t, t, lambda0)
    % Simple HHG spectrum: FFT of a_t -> S(q) normalized, q by nominal omega.
    a_t = a_t(:).';
    t   = t(:).';
    dt  = mean(diff(t));
    a_t = a_t - mean(a_t);
 
    N   = numel(a_t);
    Aom = fft(a_t).*dt;
    S   = abs(Aom).^2;
 
    f   = (0:N-1)/(N*dt);
    w   = 2*pi*f;
    c0  = 299792458;
    w0  = 2*pi*c0/lambda0;      % nominal driver frequency
    q   = w/w0;
 
    mask     = q>0;
    S        = S(mask);
    q_axis   = q(mask);
    S_norm   = S./max(S+eps);
end
 
function psi = apply_mask(psi, edge_frac)
    N = numel(psi); m = max(1, round(edge_frac*N));
    edge = 0.5*(1 - cos(pi*(0:m-1)'/m));
    mask = ones(N,1);
    mask(1:m) = edge;
    mask(end-m+1:end) = flipud(edge);
    psi = psi .* mask;
end
 
function [psi0, E0] = tdse_groundstate_itp(x_au, soft_a, varargin)
    p = inputParser; p.KeepUnmatched=true;
    addParameter(p,'dtau',0.05,@(v)isnumeric(v)&&v>0);
    addParameter(p,'max_iter',4000,@(v)isnumeric(v)&&v>=1);
    addParameter(p,'tol',1e-10,@(v)isnumeric(v)&&v>0);
    addParameter(p,'mask_edge',0.05,@(v)isnumeric(v)&&v>=0&&v<=0.5);
    addParameter(p,'report',false,@islogical);
    parse(p,varargin{:}); o = p.Results;
 
    N  = numel(x_au);
    dx = x_au(2)-x_au(1);
    dk = 2*pi/(N*dx);
    k  = fftshift((-N/2:N/2-1).' * dk);
    Vx = -1 ./ sqrt(x_au.^2 + soft_a^2);
 
    Kprop_itp = exp(-0.5*(k.^2)*o.dtau);
 
    m = max(1, round(o.mask_edge*N));
    edge = 0.5*(1 - cos(pi*(0:m-1)'/m));
    mask = ones(N,1);
    mask(1:m) = edge;
    mask(end-m+1:end) = flipud(edge);
 
    psi = (1/pi)^(1/4) * exp(-x_au.^2/2);
    psi = psi / sqrt(trapz(x_au, abs(psi).^2));
 
    E_prev = inf; E0 = inf;
    for it = 1:o.max_iter
        psi = exp(-Vx * (o.dtau/2)) .* psi;
        psi = ifft( fft(psi) .* Kprop_itp );
        psi = exp(-Vx * (o.dtau/2)) .* psi;
        if o.mask_edge>0, psi = psi .* mask; end
        psi = psi / sqrt(trapz(x_au, abs(psi).^2));
 
        psi_k = fft(psi);
        d2psi = ifft( -(k.^2) .* psi_k );
        Hpsi  = -0.5*d2psi + Vx .* psi;
        E0    = real(trapz(x_au, conj(psi).*Hpsi));
        if it > 5 && abs(E0 - E_prev) < o.tol
            if o.report, fprintf('ITP converged in %d iterations. E0=%.6f a.u.\n', it, E0); end
            break
        end
        E_prev = E0;
        if o.report && mod(it,200)==0
            fprintf('ITP iter %4d  E=%.6f a.u.\n', it, E0);
        end
    end
    psi0 = psi / sqrt(trapz(x_au, abs(psi).^2));
end
 
function print_mem_snapshot(label)
    info = whos;
    total_bytes = sum([info.bytes]);
    fprintf('  [MEM] %s: %.2f MB in workspace\n', label, total_bytes/1024^2);
end
 
function fdtd_out = fdtd_only_periodic_current(cfg, fdtd_cfg, epsr_t, J_src_mat, source_cells)
% FDTD_ONLY_PERIODIC_CURRENT
% 1D periodic FDTD with time-dependent uniform epsilon(t)
% Driven by microscopic CURRENT sources J(t,x)
% Records:
%   - E_xt(x,t)   [downsampled]
%   - Eprobe(t)
%   - E_sample_mat(t, Ns)
%   - sample_cells
% Compatible with the "step 1" field-sampling pipeline used for TDSE.
 
    %% --- constants ---
    c0   = 299792458;
    mu0  = 4*pi*1e-7;
    eps0 = 1/(mu0*c0^2);
 
    %% --- time & space ---
    t  = cfg.time.t(:).';
    dt = cfg.time.dt;
    Nt = numel(t);
 
    Nx = fdtd_cfg.Nx;
    dx = fdtd_cfg.dx;
    x  = (0:Nx-1)*dx;
 
    if numel(epsr_t) ~= Nt
        error('epsr_t must have length Nt');
    end
 
    %% --- reshape current sources ---
    if nargin < 4 || isempty(J_src_mat)
        J_src_mat = zeros(Nt,0);  % no sources
    end
    J_src_mat = reshape(J_src_mat, Nt, []);
    Ns = size(J_src_mat,2);
 
    if Ns > 0
        if nargin < 5 || numel(source_cells) ~= Ns
            error('source_cells must match number of columns in J_src_mat');
        end
        source_cells = source_cells(:).';
    else
        source_cells = [];
    end
 
    %% --- probe cell ---
    if isfield(fdtd_cfg,'probe_cell')
        probe_cell = fdtd_cfg.probe_cell;
    elseif isfield(fdtd_cfg,'probe_cell_frac')
        probe_cell = max(1, min(Nx, round(fdtd_cfg.probe_cell_frac * Nx)));
    else
        probe_cell = round(Nx/2);
    end
 
    %% --- sample cells (for TDSE sampling) ---
    if isfield(fdtd_cfg,'sample_stride_x') && ~isempty(fdtd_cfg.sample_stride_x)
        stride_x = max(1, fdtd_cfg.sample_stride_x);
        sample_cells = 1:stride_x:Nx;
    else
        sample_cells = 1:Nx;
    end
    Nsamp = numel(sample_cells);
 
    %% --- temporal stride ---
    if isfield(fdtd_cfg,'store_stride') && ~isempty(fdtd_cfg.store_stride)
        stride_t = max(1, fdtd_cfg.store_stride);
    else
        stride_t = 10;
    end
    Tframes = ceil(Nt/stride_t);
 
    %% --- allocate fields ---
    E = zeros(1,Nx);
    H = zeros(1,Nx);
 
    E_xt         = zeros(Nx, Tframes);
    Eprobe       = zeros(1,Nt);
    E_sample_mat = zeros(Nt, Nsamp);
 
    E0_snapshot = E;
 
    %% --- main loop ---
    frame_idx = 0;
 
    for n = 1:Nt
 
        % ---- H update ----
        H = H - (dt/(mu0*dx)) * (circshift(E,-1) - E);
 
        % ---- permittivity ----
        eps_abs = eps0 * epsr_t(n);
 
        % ---- build microscopic current array ----
        J_micro = zeros(1,Nx);
        if Ns > 0
            J_micro(source_cells) = J_src_mat(n,:);   % A/m^2
        end
 
        % ---- E update with current ----
        curlH = H - circshift(H,1);
        E = E - (dt/(dx*eps_abs)) * curlH - (dt/eps_abs) * J_micro;
 
        % ---- record probe ----
        Eprobe(n) = E(probe_cell);
 
        % ---- record sampled electric fields ----
        E_sample_mat(n,:) = E(sample_cells);
 
        % ---- record field snapshots ----
        if mod(n-1, stride_t) == 0
            frame_idx = frame_idx+1;
            E_xt(:,frame_idx) = E;
        end
    end
 
    %% --- trim frames ---
    E_xt = E_xt(:,1:frame_idx);
    t_store = t(1:stride_t:1+(frame_idx-1)*stride_t);
 
    %% --- output struct ---
    fdtd_out = struct();
    fdtd_out.t            = t;
    fdtd_out.dt           = dt;
    fdtd_out.x            = x;
    fdtd_out.dx           = dx;
 
    fdtd_out.E_xt         = E_xt;
    fdtd_out.t_store      = t_store;
 
    fdtd_out.Eprobe       = Eprobe;
    fdtd_out.E0_snapshot  = E0_snapshot;
 
    fdtd_out.sample_cells = sample_cells;
    fdtd_out.E_sample_mat = E_sample_mat;
 
    fdtd_out.epsr_t       = epsr_t;
    fdtd_out.probe_cell   = probe_cell;
    fdtd_out.J_sources    = J_src_mat;
    fdtd_out.source_cells = source_cells;
end
 
function fdtd_out = fdtd_only_periodic_gain(cfg, fdtd_cfg, epsr_t)
% FDTD_ONLY_PERIODIC_GAIN
% 1D periodic FDTD with time-varying but spatially uniform epsilon(t),
% initial forward eigenmode (optional), and time-dependent gain sigma(t).
%
% - Gain is implemented as a real conductivity sigma(t) (S/m):
%       eps * dE/dt + sigma * E = dH/dx
%   => E update includes factor (1 - dt*sigma/eps).
%
% - Here we also record:
%     E_xt(x,t)   [downsampled]
%     Eprobe(t)
%     E_sample_mat(t, Nsamp)  at sample_cells
%     sample_cells
%
% Usage:
%   fdtd_cfg.gain_peak_sigma  (negative for gain, S/m)
%   fdtd_cfg.gain_rise_fs     (rise duration in fs)
%   fdtd_cfg.gain_fall_fs     (total gain window length in fs)
%   If these are missing, sigma(t) = 0 (no gain).
 
    % --- constants ---
    c0   = 299792458;
    mu0  = 4*pi*1e-7;
    eps0 = 1/(mu0*c0^2);
 
    % --- time & space ---
    t  = cfg.time.t(:).';
    dt = cfg.time.dt;
    Nt = numel(t);
 
    Nx = fdtd_cfg.Nx;
    dx = fdtd_cfg.dx;
    x  = (0:Nx-1)*dx;
 
    if numel(epsr_t) ~= Nt
        error('epsr_t must have length Nt');
    end
 
    % --- build sigma(t) (S/m) from cfg, default = 0 ---
    sigma_t = zeros(1, Nt);  % default: no gain
 
    if isfield(fdtd_cfg, 'gain_peak_sigma') && fdtd_cfg.gain_peak_sigma ~= 0
        sigma_peak = fdtd_cfg.gain_peak_sigma;  % negative = gain
 
        if isfield(fdtd_cfg, 'gain_rise_fs')
            T_rise_fs = fdtd_cfg.gain_rise_fs;
        else
            T_rise_fs = 6;  % default
        end
        if isfield(fdtd_cfg, 'gain_fall_fs')
            T_fall_fs = fdtd_cfg.gain_fall_fs;
        else
            T_fall_fs = 12; % default total gain window
        end
 
        t_fs = t * 1e15;   % time in fs
 
        for n = 1:Nt
            tf = t_fs(n);
            if tf <= 0
                sigma_t(n) = 0;
            elseif tf <= T_rise_fs
                % linear ramp 0 -> sigma_peak over T_rise_fs
                sigma_t(n) = sigma_peak * (tf / T_rise_fs);
            elseif tf <= T_fall_fs
                % linear ramp sigma_peak -> 0 over (T_fall_fs - T_rise_fs)
                frac = (tf - T_rise_fs) / max(eps, (T_fall_fs - T_rise_fs));
                sigma_t(n) = sigma_peak * (1 - frac);
            else
                sigma_t(n) = 0;
            end
        end
    end
 
    % --- probe cell ---
    if isfield(fdtd_cfg,'probe_cell')
        probe_cell = fdtd_cfg.probe_cell;
    elseif isfield(fdtd_cfg,'probe_cell_frac')
        probe_cell = max(1, min(Nx, round(fdtd_cfg.probe_cell_frac * Nx)));
    else
        probe_cell = round(Nx/2);
    end
 
    % --- sample cells (for TDSE later) ---
    if isfield(fdtd_cfg,'sample_stride_x') && ~isempty(fdtd_cfg.sample_stride_x)
        stride_x = max(1, fdtd_cfg.sample_stride_x);
        sample_cells = 1:stride_x:Nx;
    else
        sample_cells = 1:Nx;
    end
    Nsamp = numel(sample_cells);
 
    % --- temporal store stride for E(x,t) ---
    if isfield(fdtd_cfg,'store_stride') && ~isempty(fdtd_cfg.store_stride)
        stride_t = max(1, fdtd_cfg.store_stride);
    else
        stride_t = 10;
    end
    Tframes = ceil(Nt/stride_t);
 
    % --- allocate fields ---
    E = zeros(1,Nx);
    H = zeros(1,Nx);
 
    E_xt         = zeros(Nx, Tframes);
    Eprobe       = zeros(1, Nt);
    E_sample_mat = zeros(Nt, Nsamp);
 
    % --- initial forward plane wave (optional) ---
    use_ic = isfield(fdtd_cfg,'ic') && isfield(fdtd_cfg.ic,'enable') && fdtd_cfg.ic.enable;
    if use_ic
        n0   = sqrt(epsr_t(1));
        k_cav = 2*pi/(Nx*dx);           % one spatial period across domain
        Smed  = (c0*dt)/(n0*dx); %#ok<NASGU>
        omega_num = (2/dt)*asin( (c0*dt/(n0*dx)) * sin((k_cav*dx)/2) );
 
        eta0    = mu0*c0;
        eta_med = eta0/n0;
 
        xE = (0:Nx-1)*dx;
        xH = ((0:Nx-1)+0.5)*dx;
 
        E0amp = fdtd_cfg.ic.E0;
        E = E0amp * cos(k_cav * xE);
        H = (E0amp/eta_med) * cos(k_cav * xH - omega_num*dt/2);
    end
 
    E0_snapshot = E;
 
    % --- main loop ---
    frame_idx = 0;
    for n = 1:Nt
        % H update (periodic)
        H = H - (dt/(mu0*dx)) * (circshift(E,-1) - E);
 
        % material epsilon
        eps_abs = eps0 * epsr_t(n);
 
        % local sigma at this time step
        sigma_n = sigma_t(n);
 
        % E update with gain/loss term
        curlH = H - circshift(H,1);
 
        % eps * (E^{n+1} - E^n)/dt + sigma*E^n = (curlH)/dx
        % => E^{n+1} = (1 - dt*sigma/eps)*E^n - (dt/(dx*eps))*curlH
        E = (1 - dt*sigma_n/eps_abs) .* E - (dt/(dx*eps_abs)) * curlH;
 
        % record probe & samples
        Eprobe(n)          = E(probe_cell);
        E_sample_mat(n,:)  = E(sample_cells);
 
        % record snapshots
        if mod(n-1, stride_t) == 0
            frame_idx = frame_idx+1;
            E_xt(:, frame_idx) = E;
        end
    end
 
    % trim unused frames
    E_xt    = E_xt(:,1:frame_idx);
    t_store = t(1:stride_t:1+(frame_idx-1)*stride_t);
 
    % pack output
    fdtd_out = struct();
    fdtd_out.t            = t;
    fdtd_out.dt           = dt;
    fdtd_out.x            = x;
    fdtd_out.dx           = dx;
 
    fdtd_out.E_xt         = E_xt;
    fdtd_out.t_store      = t_store;
 
    fdtd_out.Eprobe       = Eprobe;
    fdtd_out.E0_snapshot  = E0_snapshot;
 
    fdtd_out.sample_cells = sample_cells;
    fdtd_out.E_sample_mat = E_sample_mat;
 
    fdtd_out.epsr_t       = epsr_t;
    fdtd_out.probe_cell   = probe_cell;
 
    fdtd_out.sigma_t      = sigma_t;   % for debugging
end
