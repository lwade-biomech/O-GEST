function [Events, SpatioTemporals, Info] = T_GEST(Time, JointsDepth, Setting)
% T_GEST
% -------------------------------------------------------------------------
% Purpose:
%   Runs the O-GEST gait event detection algorithm for treadmill gait.
%   Automatically adjusts horizontal foot trajectories using estimated 
%   treadmill belt velocity.
%
% Inputs:
%   Time        - Time vector (Nx1)
%   JointsDepth - Struct containing some or all of the following fields:
%                   * Toe_L_Horizontal
%                   * Ankle_L_Horizontal
%                   * Heel_L_Horizontal
%                   * Toe_R_Horizontal
%                   * Ankle_R_Horizontal
%                   * Heel_R_Horizontal
%                   * Pelvis
%
%   Setting     - Structure containing parameters for O_GEST.
%
% Outputs:
%   Events          - Detected gait events
%   SpatioTemporals - Spatiotemporal gait parameters
%   Info            - Additional O_GEST information
% -------------------------------------------------------------------------

%% ------------------------------------------------------------------------
% Validate Inputs
% -------------------------------------------------------------------------
requiredField = "Pelvis";
allFields = fields(JointsDepth);

if ~ismember(requiredField, allFields)
    error("T_GEST:MissingPelvis", ...
        "The field 'Pelvis' is required in JointsDepth.");
end

Pelvis = JointsDepth.Pelvis;
dt     = Time(2);

%% ------------------------------------------------------------------------
% Select priority markers for treadmill velocity estimation (Heel -> Toe -> Ankle)
% -------------------------------------------------------------------------
% LEFT FOOT
if isfield(JointsDepth, 'Heel_L_Horizontal')
    velMarker_L = 'Heel_L_Horizontal';
elseif isfield(JointsDepth, 'Toe_L_Horizontal')
    velMarker_L = 'Toe_L_Horizontal';
elseif isfield(JointsDepth, 'Ankle_L_Horizontal')
    velMarker_L = 'Ankle_L_Horizontal';
else
    velMarker_L = [];
end

% RIGHT FOOT
if isfield(JointsDepth, 'Heel_R_Horizontal')
    velMarker_R = 'Heel_R_Horizontal';
elseif isfield(JointsDepth, 'Toe_R_Horizontal')
    velMarker_R = 'Toe_R_Horizontal';
elseif isfield(JointsDepth, 'Ankle_R_Horizontal')
    velMarker_R = 'Ankle_R_Horizontal';
else
    velMarker_R = [];
end

if isempty(velMarker_L) && isempty(velMarker_R)
    error("T_GEST:NoFootMarkers", "No foot markers found for velocity estimation.");
end

%% ------------------------------------------------------------------------
% Helper function: estimate treadmill velocity from a single marker and
% pevlis
% -------------------------------------------------------------------------
function vel = estimateVel(marker, Pelvis, Time)
    rel = Pelvis - marker;

    [~, HS] = findpeaks(-rel);  % heel strikes
    [~, TO] = findpeaks(rel);   % toe-offs

    if ~isempty(HS) && ~isempty(TO) && HS(1) > TO(1)
        TO = TO(2:end);
    end

    nCycles = min(numel(HS), numel(TO));
    if nCycles < 3
        vel = [];
        return
    end

    vel = zeros(nCycles,1);
    dt = Time(2);

    for i = 1:nCycles
        range  = TO(i) - HS(i);
        midIdx = round(HS(i) + range/2);

        startIdx = max(1, round(midIdx - range/4));
        endIdx   = min(length(marker), round(midIdx + range/4));

        if endIdx <= startIdx
            vel(i) = NaN;
        else
            seg = marker(startIdx:endIdx);
            g = gradient(seg, dt);
            vel(i) = -mean(g); % sign flip so belt motion is forward
        end
    end

    vel = vel(~isnan(vel));
end

%% ------------------------------------------------------------------------
% Compute treadmill velocities for available markers
% -------------------------------------------------------------------------
velocities = [];

if ~isempty(velMarker_L)
    velL = estimateVel(JointsDepth.(velMarker_L), Pelvis, Time);
    velocities = [velocities; velL];
end

if ~isempty(velMarker_R)
    velR = estimateVel(JointsDepth.(velMarker_R), Pelvis, Time);
    velocities = [velocities; velR];
end

if isempty(velocities)
    error("T_GEST:VelocityFailed", "No valid cycles found for velocity estimation.");
end

meanVel = median(velocities);        % combined single treadmill velocity
velPerFrame = meanVel * dt;
cumVel = cumsum(velPerFrame * ones(length(Time),1));

%% ------------------------------------------------------------------------
% Build Left & Right Foot Trajectories for O-GEST
% -------------------------------------------------------------------------
existingLeft  = intersect({'Heel_L_Horizontal','Toe_L_Horizontal','Ankle_L_Horizontal'}, fields(JointsDepth), 'stable');
existingRight = intersect({'Heel_R_Horizontal','Toe_R_Horizontal','Ankle_R_Horizontal'}, fields(JointsDepth), 'stable');

JointsDepth_L = [];
for k = 1:numel(existingLeft)
    fieldName = existingLeft{k};
    markerData = JointsDepth.(fieldName);
    if ~isempty(markerData)
        % Add cumVel only if the marker exists
        JointsDepth_L = [JointsDepth_L , markerData + cumVel];
    end
end

% Right side
JointsDepth_R = [];
for k = 1:numel(existingRight)
    fieldName = existingRight{k};
    markerData = JointsDepth.(fieldName);
    if ~isempty(markerData)
        JointsDepth_R = [JointsDepth_R , markerData + cumVel];
    end
end

%TEST to see if filtering helps?
%% ------------------------------------------------------------------------
% Butterworth filter helper (2nd order, 10 Hz cutoff example)
% -------------------------------------------------------------------------
function xf = bwfilt(x, fs, fc)
    [b,a] = butter(2, fc/(fs/2), 'low');
    xf = filtfilt(b, a, x);
end

%% ------------------------------------------------------------------------
% Apply Butterworth filter to treadmill-corrected trajectories
% -------------------------------------------------------------------------
if Setting.ApplyFilt == "ON"
    fs = 1/dt;     % sampling frequency
    fc = 3;       % cutoff frequency (modify as needed)
    
    [b,a] = butter(2, fc/(fs/2), 'low');
%     %TEST
%     plot(JointsDepth_L)
%     hold on
    
    if ~isempty(JointsDepth_L)
        for k = 1:size(JointsDepth_L,2)
            JointsDepth_L(:,k) = filtfilt(b, a, JointsDepth_L(:,k));
        end
    end
%     %Test
%     plot(JointsDepth_L)
    
    
    
    if ~isempty(JointsDepth_R)
        for k = 1:size(JointsDepth_R,2)
            JointsDepth_R(:,k) = filtfilt(b, a, JointsDepth_R(:,k));
        end
    end
end


%% ------------------------------------------------------------------------
% Run O-GEST
% -------------------------------------------------------------------------
[Events, SpatioTemporals, Info] = O_GEST(Time, JointsDepth_L, JointsDepth_R, Setting);

end % function T_GEST
