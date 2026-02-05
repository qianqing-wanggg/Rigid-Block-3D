import pandas as pd
direction = 'neg'
wall = "SW2"
out_file = f'../data_test/SS_02/envelop_{wall}_{direction}.csv'
out_plt = f'../data_test/SS_02/envelop_{wall}_{direction}'
df = pd.read_csv(f"../data_test/SS_02/{wall}_clean_data.csv")
df = df[['DIC_Plate_Plaster_side_mean_plate_[mm]','Horizontal_force_[kN]']]
# change headers of df
df.columns = ['top_displacement','horizontal_force']
# iterate lines
xs = []
ys = []
covered_range_min = 1
covered_range_max = -1
prev_displacement = 0
pre_force = 0
for index, row in df.iterrows():
    #print(row['top_displacement'], row['horizontal_force'])
    if row['top_displacement'] <= covered_range_max and row['top_displacement']>=covered_range_min:
        #skip this point
        continue
    elif direction=='pos' and row['top_displacement']>covered_range_max:#increasing load
        # if abs(row['horizontal_force']-pre_force)>40:
        #     continue
        xs.append(row['top_displacement'])
        ys.append(row['horizontal_force'])
        covered_range_max = row['top_displacement']
        prev_displacement = row['top_displacement']
        pre_force = row['horizontal_force']
    elif direction=='neg' and row['top_displacement']<covered_range_min:#decreasing load
        # if abs(row['horizontal_force']-pre_force)>40:
        #     continue
        xs.append(row['top_displacement'])
        ys.append(row['horizontal_force'])
        covered_range_min = row['top_displacement']
        prev_displacement = row['top_displacement']
        pre_force = row['horizontal_force']
        
#plot the envelop
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
#plot x,y axis at x = 0 and y = 0
ax.axhline(y=0, color='k',linewidth=0.5)
ax.axvline(x=0, color='k',linewidth=0.5)

#plot full simulation
ax.plot(df['top_displacement'],df['horizontal_force'],label = f'Full simulation {wall}',\
        alpha = 1,linewidth = 2,color = 'blue')
#plot envelop
ax.plot(xs,ys,'--',label = f'Test envelop {wall}',\
        alpha = 1,linewidth = 2,color = 'red')
# save plot
plt.savefig(out_plt)

# save xs,ys to a csv file
df = pd.DataFrame({'top_displacement':xs,'horizontal_force':ys})
df.to_csv(out_file,index = False)
