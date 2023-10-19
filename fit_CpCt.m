clear
clc


% This script read the .txt file containing the data for
% power coefficient and thrust coefficient and find the 
% best fit surface
%
% author: Yihan Liu
% date: 9/1/2023

% read the data 
fid = fopen('Cp_Ct.NREL5MW.txt', 'r');

% skip to the line containing pitch_angles
for i = 1:4
    fgets(fid);
end

% read data for pitch angles
pitch_angles = fscanf(fid, '%f', inf)';

fgets(fid);

% read data for TSR values
TSR_values = fscanf(fid, '%f', inf)';

% skip to the line containing the power coefficient
for i = 1:5
    fgets(fid);
end

% read data for the power coefficient
C_p = fscanf(fid, '%f', [length(pitch_angles), length(TSR_values)])';

% skip to the line containing the thrust coefficient
for i = 1:4
    fgets(fid);
end

% read data for the thrust coefficient
C_t = fscanf(fid, '%f', [length(pitch_angles), length(TSR_values)])';

% close the file
fclose(fid);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% fit the data

[x, y] = meshgrid(pitch_angles, TSR_values);
angle_data = x(:);
TSR_data = y(:);
Cp_data = C_p(:);
Ct_data = C_t(:);

ft = fittype('poly55');

% fit the power coefficient
[surfaceFitCp, gofCp] = fit([angle_data, TSR_data], Cp_data, ft);

% fit the thrust coefficient
[surfaceFitCt, gofCt] = fit([angle_data, TSR_data], Ct_data, ft);

% display results for C_p
disp('Fit parameters for C_p:');
disp(surfaceFitCp);
disp('Goodness of fit for C_p:');
disp(gofCp);

% display results for C_t
disp('Fit parameters for C_t:');
disp(surfaceFitCt);
disp('Goodness of fit for C_t:');
disp(gofCt);

% Generate mesh for visualization
[X, Y] = meshgrid(linspace(min(pitch_angles), max(pitch_angles), 100), ...
                  linspace(min(TSR_values), max(TSR_values), 100));

% Evaluate the fitted surfaces
Z_Cp = feval(surfaceFitCp, X, Y);
Z_Ct = feval(surfaceFitCt, X, Y);

% Create a new figure for Cp
figure;
hold on;
surf(X, Y, Z_Cp, 'FaceAlpha', 0.5, 'EdgeColor', 'none'); % Fitted surface
scatter3(angle_data, TSR_data, Cp_data, 'ro'); % Data points
title('Fitted Surface and Data Points for Cp');
xlabel('Pitch Angle');
ylabel('TSR');
zlabel('Cp');
hold off;
grid on;  % Enable grid
view(3);  % Set to 3D view

% Create a new figure for Ct
figure;
hold on;
surf(X, Y, Z_Ct, 'FaceAlpha', 0.5, 'EdgeColor', 'none'); % Fitted surface
scatter3(angle_data, TSR_data, Ct_data, 'bo'); % Data points
title('Fitted Surface and Data Points for Ct');
xlabel('Pitch Angle');
ylabel('TSR');
zlabel('Ct');
hold off;
grid on;  % Enable grid
view(3);  % Set to 3D view

