
## Data  

Filtered data sample should be downloaded from here (google drive link): https://drive.google.com/file/d/13WL5Qmb62nbES1qIOEk4Bm8C25cw8Mnc/view?usp=drive_link   

This dataset contains **1,253,905 historical U.S. newspaper articles published 1920–1940**, drawn from
the Library of Congress *Chronicling America* collection (via the
[`dell-research-harvard/AmericanStories`](https://huggingface.co/datasets/dell-research-harvard/AmericanStories)
OCR dataset) and filtered to those whose body or headline mentions federal-government language
(`federal`, `administration`, `government`, `new deal`, or `the feds`). Each row carries the full
OCR'd article text plus metadata, along with derived fields recording which keywords matched and how
often. Article volume is heaviest in 1920–1922 (~150K/year), thins through the mid-1920s, and rebounds
during the New Deal era; the median article runs roughly 75–80 words. Topic modeling shows the coverage
centers on legislative process, party politics and elections, fiscal policy (income tax, banking and the
Federal Reserve), and labor disputes, with politics and taxation gaining share after 1933. The text is


