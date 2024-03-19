import streamlit as st
import pandas as pd
import base64
def main():
    st.title("Excel Data Viewer and Editor")

    # File uploader
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

    if uploaded_file is not None:
        try:
            # Read Excel file
            df = pd.read_excel(uploaded_file)

            # Display DataFrame
            st.write("Data from Excel:")
            st.write(df)
            if st.button("edit sheet"):    
                # Edit the data
                st.header("Edit Data")
                edited_df = edit_data(df)
                # st.write("Edited Data:")
                # st.write(edited_df)

                # # Save the edited data to a new Excel file
                save_data(edited_df)

        except Exception as e:
            st.error(f"An error occurred: {e}")

def edit_data(df):
    # Display the DataFrame with editable cells
    # You can use Streamlit's `st.dataframe` with the `editable=True` parameter
    # edited_df = st.dataframe(df, editable=True)
    edited_df = st.data_editor(df)
    return edited_df

def save_data(df):
    # Save the edited DataFrame to a new Excel file
    # Provide a button to download the modified Excel file
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="edited_data.csv">Download edited data</a>'
    st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
