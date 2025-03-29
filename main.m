%% 数据准备（修正四元数处理）
bag = rosbag('2025-03-29-09-08-59.bag');
odom_msgs = readMessages(select(bag, 'Topic', '/aft_mapped_to_init'));

num_samples = length(odom_msgs);
global_pos = zeros(num_samples, 3);
quat_list = zeros(num_samples, 4);  % 存储格式：[w x y z]
time_stamps = zeros(num_samples, 1); % 新增时间戳存储

% 正确提取四元数（已验证格式）
start_time = odom_msgs{1}.Header.Stamp.Sec + odom_msgs{1}.Header.Stamp.Nsec*1e-9;
for i = 1:num_samples
    % 全局坐标
    global_pos(i,:) = [odom_msgs{i}.Pose.Pose.Position.X, 
                      odom_msgs{i}.Pose.Pose.Position.Y,
                      odom_msgs{i}.Pose.Pose.Position.Z];
    
    % 四元数转换（ROS→MATLAB）
    quat_raw = [odom_msgs{i}.Pose.Pose.Orientation.W,   % w
                odom_msgs{i}.Pose.Pose.Orientation.X,   % x
                odom_msgs{i}.Pose.Pose.Orientation.Y,   % y 
                odom_msgs{i}.Pose.Pose.Orientation.Z];  % z
    quat_list(i,:) = quat_raw / norm(quat_raw);
    
    % 记录时间戳（新增）
    stamp = odom_msgs{i}.Header.Stamp;
    time_stamps(i) = (stamp.Sec + stamp.Nsec*1e-9) - start_time;
end

%% 非线性优化建模
% 初始猜测（基于几何中心）
C0 = mean(global_pos, 1)';             % 3x1
P0 = [0.1; 0.1; 0.1];                 % 假设初始偏移

% 定义目标函数
cost_func = @(params) objective(params, global_pos, quat_list);

% 配置优化器
options = optimoptions('fmincon',...
    'Algorithm','interior-point',...
    'MaxIterations', 1000,...
    'MaxFunctionEvaluations', 1e4,...
    'Display','iter-detailed',...
    'UseParallel',true);

% 执行优化
solution = fmincon(cost_func, [C0; P0], [], [], [], [], [], [], [], options);

% 解析结果
C_opt = solution(1:3)';
P_local_opt = solution(4:6)';

%% 验证与分析
% 计算优化后的中心误差
center_errors = zeros(num_samples,3);
for i = 1:num_samples
    R = quat2rotm(quat_list(i,:));
    C_calculated = global_pos(i,:)' - R * P_local_opt';
    center_errors(i,:) = C_calculated' - C_opt;
end

% 可视化误差分布
figure
subplot(2,1,1)
plot(vecnorm(center_errors,2,2))
title('优化后中心误差模长')
xlabel('样本序号'); ylabel('误差(m)')

subplot(2,1,2)
histogram(vecnorm(center_errors,2,2), 50)
title('误差分布直方图')
xlabel('误差值(m)'); ylabel('频次')

% 打印统计结果
fprintf('\n=== 优化结果 ===\n');
fprintf('系统中心(全局): [%.6f, %.6f, %.6f]\n', C_opt);
fprintf('局部坐标:       [%.6f, %.6f, %.6f]\n', P_local_opt);
fprintf('最大中心误差:   %.6f m\n', max(vecnorm(center_errors,2,2)));
fprintf('平均误差:       %.6f m\n', mean(vecnorm(center_errors,2,2)));

%% 新增：动态系统中心计算（必须添加此部分）
% 计算每个时刻的理论系统中心
dynamic_centers = zeros(num_samples, 3);
for i = 1:num_samples
    R = quat2rotm(quat_list(i,:));            % 当前时刻旋转矩阵
    dynamic_centers(i,:) = global_pos(i,:) - (R * P_local_opt')';
end

%% 核心可视化：原始数据与动态系统中心对比
figure('Position', [100 100 1200 800])

% 三维空间分布
subplot(2,2,[1,3])
scatter3(global_pos(:,1), global_pos(:,2), global_pos(:,3), 30, 'b', 'filled')
hold on
scatter3(dynamic_centers(:,1), dynamic_centers(:,2), dynamic_centers(:,3), 50, time_stamps, 'filled')
plot3(C_opt(1), C_opt(2), C_opt(3), 'rp', 'MarkerSize', 20, 'MarkerFaceColor','r')
title('三维空间分布对比')
xlabel('X_{global}'); ylabel('Y_{global}'); zlabel('Z_{global}')
legend({'原始数据','动态计算中心','优化中心'}, 'Location','best')
colormap(jet)
colorbar('Ticks',linspace(min(time_stamps),max(time_stamps),5),...
         'TickLabels',compose('%.1fs',linspace(min(time_stamps),max(time_stamps),5)))
grid on; axis equal; view(-30,30)

% 动态中心时序分析
subplot(2,2,2)
plot(time_stamps, dynamic_centers(:,1), 'b', 'LineWidth', 1.5)
hold on
plot(time_stamps, dynamic_centers(:,2), 'g', 'LineWidth', 1.5)
plot(time_stamps, dynamic_centers(:,3), 'm', 'LineWidth', 1.5)
yline(C_opt(1), 'b--', 'LineWidth', 1.5)
yline(C_opt(2), 'g--', 'LineWidth', 1.5)
yline(C_opt(3), 'm--', 'LineWidth', 1.5)
title('动态系统中心坐标时序')
xlabel('时间 (s)'); ylabel('坐标值 (m)')
legend({'X动态','Y动态','Z动态','X优化','Y优化','Z优化'}, 'Location','best')
grid on

% 动态中心误差分布
subplot(2,2,4)
boxchart([abs(dynamic_centers(:,1)-C_opt(1)),...
          abs(dynamic_centers(:,2)-C_opt(2)),...
          abs(dynamic_centers(:,3)-C_opt(3))])
title('坐标轴误差分布')
set(gca,'XTickLabel',{'X','Y','Z'})
ylabel('绝对误差 (m)')
grid on

%% 动态系统中心稳定性分析
figure('Position', [100 100 800 400])

% 中心运动轨迹
subplot(1,2,1)
plot3(dynamic_centers(:,1), dynamic_centers(:,2), dynamic_centers(:,3), 'b-')
hold on
plot3(C_opt(1), C_opt(2), C_opt(3), 'rp', 'MarkerSize', 15)
title('系统中心运动轨迹')
xlabel('X'); ylabel('Y'); zlabel('Z')
grid on; axis equal; view(-30,30)

% 动态中心统计
subplot(1,2,2)
cov_matrix = cov(dynamic_centers);
heatmap({'X','Y','Z'}, {'X','Y','Z'}, cov_matrix,...
         'Colormap',parula,'ColorLimits',[0 max(cov_matrix(:))])
title('坐标协方差矩阵')

%% 误差量化分析
fprintf('\n=== 动态系统中心分析结果 ===\n');
fprintf('坐标波动标准差:\n  X: %.6f m\n  Y: %.6f m\n  Z: %.6f m\n',...
        std(dynamic_centers));
fprintf('最大坐标偏移:\n  X: %.6f m\n  Y: %.6f m\n  Z: %.6f m\n',...
        max(abs(dynamic_centers - C_opt)));

%% 目标函数定义
function total_cost = objective(params, global_pos, quat_list)
    C = params(1:3);          % 系统中心 (3x1)
    P_local = params(4:6);    % 局部坐标 (3x1)
    
    total_cost = 0;
    for i = 1:size(global_pos,1)
        % 当前旋转矩阵
        R = quat2rotm(quat_list(i,:));
        
        % 计算理论中心
        C_predicted = global_pos(i,:)' - R * P_local;
        
        % 累积误差
        error = C - C_predicted;
        total_cost = total_cost + sum(error.^2);
    end
end