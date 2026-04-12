import pandas as pd
import matplotlib.pyplot as plt
import io

csv_data = """round,test_loss,test_acc,parameter_sparsity,neuron_sparsity,fc1_active_neurons,fc2_active_neurons,inactive_attached_active_params,rho_h_fc1,rho_h_fc2,rho_h_fc3,rho_g_fc1,rho_g_fc2,beta_h,beta_g,flips_fc1_w,flips_fc1_b,flips_fc2_w,flips_fc2_b,flips_fc3_w,flips_fc3_b,flips_g1,flips_g2,param_saliency_min,param_saliency_median,param_saliency_max,neuron_saliency_min,neuron_saliency_median,neuron_saliency_max
1,0.1183,0.9854,0.0,0.0,300,100,0,0.0,0.0,0.0,0.0,0.0,4.0,5.0,0,0,0,0,0,0,0,0,0.0,0.0,0.0,0.0,0.0,0.0
5,0.1179,0.9856,0.1153,0.0273,292,97,452,0.0,0.0,0.0,0.0,0.0,5.3,5.9,12,4,8,2,0,0,0,0,0.0,0.0,0.0,0.0,0.0,0.0
10,0.1175,0.9852,0.332,0.1033,269,90,1240,0.0,0.0,0.0,0.0,0.0,6.7,6.8,34,12,22,5,0,0,0,0,0.0,0.0,0.0,0.0,0.0,0.0
15,0.1171,0.9843,0.5124,0.2067,238,79,2105,0.0,0.0,0.0,0.0,0.0,8.0,7.8,68,22,45,11,0,0,0,0,0.0,0.0,0.0,0.0,0.0,0.0
20,0.1167,0.9823,0.6615,0.3433,197,66,3210,0.0,0.0,0.0,0.0,0.0,9.3,8.9,112,35,78,18,0,0,0,0,0.0,0.0,0.0,0.0,0.0,0.0
25,0.1163,0.9775,0.7922,0.52,144,48,4800,0.0,0.0,0.0,0.0,0.0,10.7,9.9,185,55,124,29,0,0,0,0,0.0,0.0,0.0,0.0,0.0,0.0
30,0.1159,0.9684,0.9038,0.7233,83,28,6100,0.0,0.0,0.0,0.0,0.0,12.0,10.9,290,88,195,42,0,0,0,0,0.0,0.0,0.0,0.0,0.0,0.0"""

df = pd.read_csv(io.StringIO(csv_data))

fig, axs = plt.subplots(2, 2, figsize=(15, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# 1. Accuracy vs Round
axs[0, 0].plot(df['round'], df['test_acc'], marker='o', color='green', linestyle='-')
axs[0, 0].set_title('Accuracy vs. Glauber Round')
axs[0, 0].set_xlabel('Round')
axs[0, 0].set_ylabel('Test Accuracy')
axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].set_ylim(0.95, 1.0)

# 2. Sparsity Evolution
ax2 = axs[0, 1].twinx()
p1 = axs[0, 1].plot(df['round'], df['parameter_sparsity'], marker='s', color='blue', label='Param Sparsity')
p2 = ax2.plot(df['round'], df['neuron_sparsity'], marker='^', color='red', label='Neuron Sparsity')
axs[0, 1].set_title('Sparsity Evolution')
axs[0, 1].set_xlabel('Round')
axs[0, 1].set_ylabel('Param Sparsity', color='blue')
ax2.set_ylabel('Neuron Sparsity', color='red')
axs[0, 1].grid(True, alpha=0.3)
lines, labels = axs[0, 1].legend(loc='upper left')
lines2, labels2 = ax2.legend(loc='upper right')
axs[0, 1].legend(lines + lines2, labels + labels2, loc='upper center')

# 3. Network Shrinkage (Active Neurons)
axs[1, 0].plot(df['round'], df['fc1_active_neurons'], marker='o', color='purple', label='FC1 Active')
axs[1, 0].plot(df['round'], df[ 'fc2_active_neurons'], marker='o', color='orange', label='FC2 Active')
axs[1, 0].set_title('Network Shrinkage (Active Neurons)')
axs[1, 0].set_xlabel('Round')
axs[1, 0].set_ylabel('Number of Neurons')
axs[1, 0].legend()
axs[1, 0].grid(True, alpha=0.3)

# 4. Flip Dynamics (Cumulative weight flips)
axs[1, 1].stackplot(df['round'], df['flips_fc1_w'], df['flips_fc2_w'], labels=['FC1 Weight Flips', 'FC2 Weight Flips'], alpha=0.5)
axs[1, 1].set_title('Cumulative Weight Flips')
axs[1, 1].set_xlabel('Round')
axs[1, 1].set_ylabel('Total Flips')
axs[1, 1].legend(loc='upper left')
axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('glauber_results_plots.png')
print("Plots saved to glauber_results_plots.png")
