import re
import matplotlib.pyplot as plt
log_file ="src/GANFinal_loss.txt" #path to the log file containing loss values
def extract_losses(log_file: str):
    
    
    train_loss= []
    

    with open(log_file, "r") as f:
        for line in f:
            print(line) #print each line of the log file to help debug and verify that the correct lines are being read and parsed for loss values
            if "Train Loss" in line:
                train_match = re.search(r"Train Loss = ([\d.]+)", line) #regex to find training L1 loss values
                if train_match:
                    train_loss.append(float(train_match.group(1))) #append the extracted training L1 loss value to the list
          

           
    return train_loss #return the lists of training  losses


train_loss = extract_losses(log_file) #call the function to extract losses from the log file
epochs = range(1, len(train_loss) + 1) #create a range of epoch numbers based on the number of loss values extracted
plt.figure(figsize=(10, 5)) #set the figure size for the plot
plt.plot(epochs, train_loss, label="Training Loss (L1 + Adversarial)") #plot training  loss against epochs
plt.xlabel("Epochs") #label for x-axis
plt.ylabel("Training Loss") #label for y-axis
plt.title("Training Adversarial Loss Curve for GAN") #title for the plot
plt.legend() #add legend to the plot
plt.grid() #add grid to the plot for better readability
plt.savefig("GAN_TrainingLoss_Curves.png") #save the plot as an image file
plt.show() #display the plot
    