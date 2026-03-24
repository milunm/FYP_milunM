import re
import matplotlib.pyplot as plt
log_file ="src/GANFinal_loss.txt" #path to the log file containing loss values
def extract_losses(log_file: str):
    
    
    train_l1 = []
    val_l1 = []

    with open(log_file, "r") as f:
        for line in f:
            print(line) #print each line of the log file to help debug and verify that the correct lines are being read and parsed for loss values
            if "Train Loss" in line:
                train_match = re.search(r" L1 Loss = ([\d.]+)", line) #regex to find training L1 loss values
                if train_match:
                    train_l1.append(float(train_match.group(1))) #append the extracted training L1 loss value to the list
            elif "Val L1 Loss" in line:
                val_match = re.search(r"Val L1 Loss = ([\d.]+)", line) #regex to find validation L1 loss values
                if val_match:
                    val_l1.append(float(val_match.group(1))) #append the extracted validation L1 loss value to the list

           
    return train_l1, val_l1 #return the lists of training and validation L1 losses


train_l1, val_l1 = extract_losses(log_file) #call the function to extract losses from the log file
epochs = range(1, len(train_l1) + 1) #create a range of epoch numbers based on the number of loss values extracted
plt.figure(figsize=(10, 5)) #set the figure size for the plot
plt.plot(epochs, train_l1, label="Train L1 Loss") #plot training L1 loss against epochs
plt.plot(epochs, val_l1, label="Validation L1 Loss") #plot validation L1 loss against epochs
plt.xlabel("Epochs") #label for x-axis
plt.ylabel("L1 Loss") #label for y-axis
plt.title("Training and Validation L1 Loss Curves for GAN") #title for the plot
plt.legend() #add legend to the plot
plt.grid() #add grid to the plot for better readability
plt.savefig("GAN_Loss_Curves.png") #save the plot as an image file
plt.show() #display the plot
    

     

