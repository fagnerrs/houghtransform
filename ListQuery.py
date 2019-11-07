listt = [{
      "CustomerId": "1",
      "Date": "2017-02-02",
       "Content": "AAAAAAAA",
      "Type": 2

    },
    {
       "CustomerId": "2",
       "Date": "2017-02-03",
      "Content": "BBBBBBBB",
       "Type": 6
     },
     {
      "CustomerId": "3",
      "Date": "2017-02-01",
       "Content": "CCCCCCCCC",      "Type": 1
   },
     {
       "CustomerId": "4",
       "Date": "2017-02-12",
       "Content": "DDDDDDDDDD",
      "Type": 2
    }, ]

print(max(d['Date'] for d in listt if d['Type'] == 1))
print(max([d for d in listt], key=lambda x: x['Type']))