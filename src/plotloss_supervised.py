import re
import matplotlib.pyplot as plt
log_file ="CNNFinal_loss_progression.txt" #path to the log file containing loss values
def extract_losses(log_file: str):
    
    
    train_l1 = []
    val_l1 = []

    with open(log_file, "r") as f:
        for line in f:
            print(line) #print each line of the log file to help debug and verify that the correct lines are being read and parsed for loss values
            match = re.search(
                r"train L1:\s*([\d.]+)\s*\|\s*val L1:\s*([\d.]+)",
                line
                ) #regex to find training and validation L1 loss values
            if match:
                train_l1.append(float(match.group(1)))
                val_l1.append(float(match.group(2)))

    return train_l1, val_l1 #return the lists of training and validation L1 losses


train_l1, val_l1 = extract_losses(log_file) #call the function to extract losses from the log file
epochs = range(1, len(train_l1) + 1) #create a range of epoch numbers based on the number of loss values extracted
plt.figure(figsize=(10, 5)) #set the figure size for the plot
plt.plot(epochs, train_l1, label="Train L1 Loss") #plot training L1 loss against epochs
plt.plot(epochs, val_l1, label="Validation L1 Loss") #plot validation L1 loss against epochs
plt.xlabel("Epochs") #label for x-axis
plt.ylabel("L1 Loss") #label for y-axis
plt.title("Training and Validation L1 Loss Curves for Baseline CNN") #title for the plot
plt.legend() #add legend to the plot
plt.grid() #add grid to the plot for better readability
plt.savefig("CNN_Loss_Curves.png") #save the plot as an image file
plt.show() #display the plot
    

     
