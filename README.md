# VisAlign

requires  pyralysis:

https://gitlab.com/miguelcarcamov/pyralysis

# arguments:

datacolumn='DATA',  # data column for input 
datacolumns_output='DATA',  # data column to punch shifted vis. data

note the input ms will be copied to the output ms, and then overwritten - if there is a `corrected' data column and `data' is chosen for the output, the shifted `corrected' will be punched into `data', and `corrected' will be the original. Best to use either both `data' or both `corrected'. 

fit for error bars in both coordinates from
https://aip.scitation.org/doi/pdf/10.1063/1.4823074
