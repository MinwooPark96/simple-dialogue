!pip install transformers==4.45.1 -U

# Download the DailyDialog dataset
!wget https://www.dropbox.com/scl/fi/ai4je7bp3difjeuyk3a8s/dailydialog.json?rlkey=fe6pm2iz7nsb5fwulsjch5qzy -O dailydialog.json

# Download a trained model (max_data_size=100000, n_epochs=2)
!wget https://www.dropbox.com/scl/fi/ykmal3w44vqz8brd0aowk/model_checkpoint.pt?rlkey=51lox1vvyu9r07tbkfz3jrvr4 -O model_checkpoint.pt