import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ticker = pd.read_json(r'E:\02_Document\08_Python_Exercise\company_tickers.json').transpose()
# Here is the location of my document.
# Replace the location between the single quotes ('') with the path to your document, keeping the r prefix.

def fuzzy_search_and_select(df):
    while True:
        search_term = input("Enter keyword: ")
        # Search
        filtered_names = df[df['ticker'].str.contains(search_term, case=False, na=False) |
                            df['title'].str.contains(search_term, case=False, na=False)]

        # if no result match
        if filtered_names.empty:
            print("No correspond results。")
            retry = input("Restrat？press 'Y'，Press anywhere to quit: ")
            if retry.upper() == 'Y':
                continue  # restart to search
            else:
                return None
        else:
            # show the result
            print("Result：")
            print(filtered_names)

            # user choose the index
            try:
                selected_index = int(input("Select the index number (e.g., 0, 1, 2): "))
                if selected_index in filtered_names.index:
                    selected_row = filtered_names.loc[selected_index]
                    print(f"\nYou selected: \n{selected_row}")
                    return selected_row
                else:
                    print("This number is invalid。")
            except ValueError:
                print("Please enter the right row number。")


selected_row = fuzzy_search_and_select(ticker)

if selected_row is not None:
    selected_ticker = selected_row.loc['ticker']
    start = input('enter the start date (YYYY-MM-DD)：')
    end = input('enter the end date (YYYY-MM-DD)：')
    if not end:
        end = datetime.now().strftime('%Y-%m-%d')
        print(f"The end date defaults to today's date: {end}")

    # download the info on yahoo finance
    stock_data = yf.download(selected_ticker, start, end)
    stock_data_reset = stock_data.reset_index()
    print(stock_data_reset)

    while True:
        print(f'Which information you want between the following?:\n"Open   High   Low   Close   Adj Close   Volume "')
        info_list = []
        input_prompts = [
            "Please select the 1 name（Mandatory）：",
            "Please select the 2 name（Option）：",
            "Please select the 3 name（Option）：",
            "Please select the 4 name（Option）：",
            "Please select the 5 name（Option））："
        ]

        for prompt in input_prompts:
            user_input = input(prompt)
            if not user_input:
                break
            else:
                info_list.append(user_input)

        # check if user choose the good index
        available_columns = [col for col in info_list if col in stock_data_reset.columns]

        if available_columns:
            fig, axes = plt.subplots(len(available_columns), 1, figsize=(10, 5 * len(available_columns)))

            if len(available_columns) == 1:
                axes = [axes]

            for i, col in enumerate(available_columns):
                axes[i].plot(stock_data_reset['Date'], stock_data_reset[col], label=col, color=np.random.rand(3, ))
                axes[i].set_title(f"{selected_ticker} {col} Over Time", fontsize=16, fontweight='bold')
                axes[i].set_xlabel('Date', fontsize=12)
                axes[i].set_ylabel(col, fontsize=12)
                axes[i].legend(fontsize=12)
                axes[i].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

                axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                axes[i].tick_params(axis='x', rotation=45)

            plt.tight_layout(pad=2.0, h_pad=3)
            plt.subplots_adjust(top=0.92)
            plt.show()
            break
        else:
            print(f"The name is invalid, Select between: {', '.join(stock_data_reset.columns)}")
            retry_choice = input("Reselect the name？Press 'Y'，Press anywhere to quit: ")
            if retry_choice.upper() != 'Y':
                print("Quit this programme")
                break
else:
    print("No valid row is selected, exit the program。")
