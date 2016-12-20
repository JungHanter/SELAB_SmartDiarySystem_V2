from diary_analyzer import tagger
# http://www.traveldiariesapp.com/en/PublicDiary/f8623aa8-4e80-4026-9c3d-13cdf7d41a2d/Chapter/a281c0e3-ef5e-407d-9809-bc28bdbcc373
diaries = [
"""Hi Arrived in Brisbane on Thursday 3rd September. Pretty tired but watched at least 8 films on the flight!! Set off from Brisbane on Friday the 4th aboard White Spirit which can only be described as a luxury boat. It is a 55 feet long catamaran (Power Boat) and has every conceivable facility onboard, two independent cabins with bathrooms, dining room (outside and inside), TV lounge, full internet access, as well as a good looking Captain! First stop Bribie Island, anchored up and had a great meal.""",
"""Sailed From Bribie Island to Mooloolaba. Anchored up in the bay and met some of Rick and Lynne's friends who have just moved into the area. Walked along the surf beach and lunched at The Hogs Breath Cafe (not as bad as it sounds).""",
"""Sailed from Mooloolaba up to Wide Bay Bar, a 7 hour cruise on the Coral Sea. The sea was quite kind considering the scary stories told to us by Lynne! Finished up in Pelican Bay. Viv went for a run on the beach and I went for a good walk, both really to get our land legs back. Dined on board eating salmon. Map across page. Route is from Brisbane to Lady Musgrave Island.""",
"""Sailed from Pelican Point to Tin Can Bay. Anchored up and went ashore. Lunched on Parrot fish and wine, very cool. Had a stroll into Tin Can Bay. A real holiday spot with a few shops and loads of beach.""",
"""Went ashore and watched a Dolphin feeding session, very entertaining. Breakfasted on the quay, weighed anchor, and set off North up the Great Sandy Strait. The Strait runs up the inside of Fraser Island, the largest sand Island in the world (about 30 miles long). Heading for Garry's Landing.""",
"""Arrived at Garry's Landing, a short haul from Tin Can Bay, still on the Great Sandy Strait. Fraser Island is a National Park and the largest sand island in the world. We went ashore on Fraser Island and walked the bush trails. The Island is totally unspoilt and there was plenty of wild life including Dingos, Koala Bears and Sea Cows living on the lush vegetation in the mangroves. The weather has been glorious but there are strong winds predicted, so after tonight's stay we may elect to remain here for another day.""",
"""Weather glorious so didn't remain at Garry' Landing and sailed up the Great Sandy Strait and anchored in Kingfisher Bay. A beautiful spot. There is an Eco Resort just inland and we had an excellent lunch there, followed by a walk along the beach for few miles. Back to the boat and Rick had caught a Mud Crab in one of his crab pots. Should be a good lunch tomorrow.""",
"""Sailed four about 6 hours and arrived in a sheltered estuary at Bundaberg. Bundaberg is the centre of the sugar cane industry and famous Bundaberg Rum. Had a walk on shore and went out for dinner in the evening at the local Yacht Club.""",
"""Sailed for 5 hrs with a 30 knot tailwind. Fairly rough but great journey. Had to leave early, about 5 o'clock to make sure we met the tide at 1770. Arrived at the 1770 anchorage through a very narrow channel, which is only accessible at mid tide or higher. As we turned into the bay we had to cross the prevailing wind which made the access to the bay tricky but we made it and suddenly we were in very calm water. 1770 is named after Captain Cook who made a landing here on the 24th May 1770. Further round the bay is Bustard Head where Captain Cook shot a 17 lb Bustard and provided fresh meat for his hungry crew. We are now about 270 miles north of Brisbane. Went ashore for a good walk and returned to the boat to make ready for the Birthday Dinner (see picture). After Champagne we had barbecued belly pork with all the trimmings, plenty of wine and a home made Creme Brulee, all on the boat!! Opened the cards from home and read the messages on the net. We stayed for the night and will remain in the bay for another day until the weather settles before the run to Lady Musgrave Island.""",
"""Stayed in 1770 Bay for the day. Went ashore and walked past Cooks Monument to Round Hill Point. Lunched at the local Bistro on Red Emperor Reef Fish. Very tasty. Weather looking good for tomorrow when we will be sailing to Pancake Creek, anchoring up and getting ready for the sprint to Lady Musgrave Island on Monday/Tuesday.""",
"""Stayed anchored in Pancake Creek for the day. Went ashore and had a good walk through the bush over the hill to a look out point overlooking Bustard Bay, which is beautiful and unspoilt. Met some of Rick and Lynne's friends, Mike & Sylvia, who were also anchored up in the Bay and they came over in their tender for cheese and wine. Sylvia was originally from Chorlton-cum- Hardy and hadn't met anybody before who knew of such a place!! The lighthouse below was brought originally from Scotland in pieces of mainly cast iron, carried up the hill from Jenny Lin's Creek and reassembled on the rocks above. It was bought and maintained by an enthusiast.""",
"""Another beautiful day. Went ashore and walked through the bush to the lighthouse on Bustard Head. The lighthouse is defunct but has been completely restored as a museum and a tribute to the Lighthouse Keepers of old. We were given a tour and tea, crackers and cheese which was a nice gesture. Amazing tales of the past!! Then Brian went fishing with Rick. Took the RIB out into the ocean and trawled for Mackrel but to no avail. Good job we've got plenty of food on board.""",
"""Weather was fair so we set off for Lady Musgrave Island at 5am. This involved sailing east across the prevailing currents. At first everything was OK but the rolling and pitching caught up and I had to go below and hide. The trip took about 4 hours and as we approached the reef you could see the island in the lagoon. Very pretty. The reef is about 10 miles in circumference and the lagoon is about 3 miles across. The entrance through the reef is really narrow but once inside the lagoon was extremely calm. We anchored up and went ashore. Walked a lap of the island, about 1.25 miles, all on coral, returned to boat and went for a swim off the back in the lagoon. Another great meal and to bed.""",
"""Woke to a stunning dawn and breakfasted on fresh Parrot Fish caught by Rick on one of his fishing trips outside the reef. Returned to the Island and explored the interior as well as more inspection of the coral. The Island is a sanctuary for birds both for stopping off on their migrations and nesting in the trees. Lots of turtles , crabs and fish around the shore and thousands of shells lying on the coral. The forest is lovely and there were tracks through the trees to see more of the interior. Back to the boat and trips in the dinghy using glass bottom tubes to see the reefs underwater. Spectacular views of reef fish, turtles and squid! More swimming and then another wonderful meal on board, roast lamb with all the trimmings.""",
"""Weather is suitable for a run south back to Hervey Island. So we are off on a 12 hour sail but the sea is calm and we now have our sea legs! On route we saw one whale but the highlight was 6 dolphins who swam at the bow of the boat for over an hour. We were travelling at 10 knots, 12 mph, and the Dolphins were effortlessly maintaining that speed and shooting ahead at times. They are very playful and love showing off either individually or in pairs. Managed to get some good photos but they are on the camera so you will have to wait. After 10+ hours we arrived at the north end of Hervey Bay and anchored up outside Scarness and Torquay!! Dined on reef fish caught by Rick outside the reef at Lady Musgrave. The weather is forecast to blow strongly from the south so we may have to stay here for a couple of days.""",
"""Went ashore to Scarness and walked the coast. After lunch we visited the local Museum which was very interesting. It covers 3 acres and 8,000 exhibits with a couple of live displays of rope making and a blacksmith at work. There were all the buildings you might expect, church, cooper, school, milking shed, women's clothes of the time, shop etc., all really well set up. The whole task has been done by local volunteers and they obviously retain the skills of their forefathers. Dinner was on board with the 'Michelin Chef', Lynne, including Lamb Shanks and fresh veg.""",
"""Still anchored in the Bay waiting for a shift in the wind. Went ashore, Viv ran her usual distance and I walked for the same time. Had lunch in the local pub. Dined on Red Emperor fish, very good. In the afternoon another walk along the beach to Torquay which is a very popular holiday area with shallow waters in the Bay for great family swimming. Hervey Bay is one of the best areas for watching for Hump Back Whales and we hope to see them once we head down the Great Sandy Strait.""",
"""Set off at 6am and sailed south into the Great Sandy Strait. Had a very comfortable run down into Kingfisher Bay. Anchored up and went ashore for a good walk down the beach. After lunch we relaxed with a few glasses of wine, some music and enjoyed a really beautiful day with the addition of a wedding on the shore! Very peaceful evening with another splendid meal on board.""",
"""Left Kingfisher Bay early to travel as far as we can through the Great Sandy Strait today, ready for the long trek (14 hours) tomorrow back to Brisbane. Incredibly it is raining quite heavily (just like home), the first rain have seen, but it is still very warm, about 23 deg. Decided to anchor up for the night in Pelican Bay and managed a good walk through the bush. The weather cleared in the afternoon and it was very warm with a glorious sunset.""",
"""The weather forecast for Wednesday on is for very strong winds (40 mph), so we left at first light, 5am, and sailed south for 13 hrs, 140 miles, and arrived in Brisbane at 6pm. The sailing for the first couple of hours was quite tough but once we turned to travel due south it was pretty calm. In Moreton Bay we suddenly met a pod of whales, six in fact including s couple of calves. Fabulous sight and managed to get a couple of photos.""",
"""A little tired today so only rose at 6am! Took the 'Cat' (Catamaran Taxi) into Brisbane and mooched around the shops, had lunch in Myers, the equivalent of John Lewis and got the Cat back to base. Been invited out for dinner tonight, to friends of Rick & Lynne. Great company with Tim & Elaine, and another lovely meal. Didn't get back until 11 o'clock which is late by Aussie standards but we will be up early tomorrow.""",
"""Got up at 6am and went for a good walk. Checked out the 50M University pool which is open from 4.30am until 8pm every day to all. After breakfast we went into Brisbane again and spent several hours in the Brisbane Art Gallery, which is a very spacious modern building. Lots of interesting art, particularly Aborigine art. There was also a travelling Australian Photographic Exhibition, with photos back to the early 1840's onwards.""",
"""Managed to get out for a bike ride and Viv went for a run at 6.30am! Weather gorgeous so went for a coffee at a local mall. Then went for lunch locally with Rick. In the evening Rick and Lynne hosted a dinner with another two of their friends, Herb and Snorter and Darby, Lynne's niece and her friend Jessie. Very good meal and we watched the semi final of the Rugby League Cup which the Brisbane Broncos won convincingly against the odds.""",
"""Went for a swim in the 50m University of Queensland pool, outdoor naturally, at 7.30am! Held our own against the Aussies!!! In the afternoon we went to Bruce and Rhonda's for lunch and dinner. We had prawns, oysters and Moreton Bay Bugs, which were very large and very tasty, quite like crab but more succulent. More wine and steak, got back to base in time to watch the second semi final of the Rugby League Cup. Good game won convincingly by the Cowboys. Final next Sunday.""",
"""Another run and bike ride at 6am! Not quite so busy today, even Aussies have some rest on Sundays. Rick took us up to Mount Cooltha, which is the highest point around Brisbane with great views and a very clear day. Lunch was organised at a local restaurant with Rick's daughter Tammy, her husband Anthony and their 5 year old son Cooper, who is heavily into Star Wars Lego! We had planned a walk after lunch but a rainstorm came in and we had to abandon the idea. On the news tonight it showed parts of Brisbane were hit by a hailstorm with large hailstones and suffered 50mm of rain in 30 mins.""",
"""Went for an early walk, at 6am, had breakfast and went to the Botanic Gardens. The gardens are laid out over 3 acres and cover a number of areas of plant life. The best area from our point of view was the Australian flora, examples, Macademia Nut plant, types of Eucalyptus tree, Palms and Figs trees. After 3 hours in the Gardens we grabbed some lunch and boarded a bus to downtown Brisbane and visited the art area, galleries, book stores etc. In the evening we visited the local fish restaurant for local fish, calamari, and scallops!!""",
"""Took an early flight to Norfolk Island. On the map it looks very close to mainland Australia but in fact it was a 2hr flight and about 1,000 miles out in the Pacific. The big difference with Queensland was that the Island has a sub-tropical climate, is very green and is very British. Also the economy is not taxed which is a sore point with the mainland. Arrived in the early afternoon with a 1.5 hour time gain. We had booked an excellent cottage just outside 'Burnt Pine' (Local Town). Had some lunch and picked up provisions from the local supermarket. Also booked a number of trips-tours for the next 4 days. More information tomorrow.""",
"""Norfolk Island is just unique and very lovely. The people pay no taxes, the crime rate is nil and everything that they have is brought in by sea and offloaded via lighters because there is no deep water quay. They are a close knit community of about 1200 people and the one shopping area is at the Town of Burnt Pine. Today we spent the morning around the old penal colony. The main buildings are still there and have been maintained using the same materials and colours. Four of the buildings have been converted into museums and we will be going on a guided tour of these museums on Friday afternoon. There is a beautiful bay called Emily Bay (named after the Engineers wife who loved swimming in the 1850's). It is great for swimming in the Pacific. This evening we joined a group for a progressive dinner. This involved a starter at one house, the main course at a second house and the dessert at a third house. This a very common process and you meet a lot of people, mainly Australians, who have all visited the UK, but nor seen very much of it!! The meal was great but very British and the hosts at each house were very welcoming, pleasant and interesting.""",
"""It is an amazing Island. There are no predatory animals so wild chickens run freely on the verges. This morning we went for a walk along the cliffs and back through the golf course which surrounds the old Governers House. Later we drove to the monument to Captain Cook, who landed there in 1774. The actual spot is shown and is spectacular with lots of rocks. They must have had nerves of steel to sail in these waters. The main reason for Cook to land, was to see if Norfolk Pines could be used to make main masts. However they were not usable because they had many knots and were therefore weak. Then on to Anson Bay, another beautiful spot, where the British Navy had the idea to harvest the native flax plants to make ropes and sails. This evening we went to an open air Fish Fry on a headland overlooking the sea, with loads of food and fresh fish, just great! As the feast continued there was a local singer originally from Pitcairn Island, and four local girls, who demonstrated Tahitian Dancing, all quite excellent. In fact all the girls and the singer were descendants of the Bounty mutineers. Next there was an open air enactment of the Bounty Mutiny, which had a really spectacular set and was very professionally acted with all of the actors again, descendants of the mutineers. We are now very knowledgeable about the Bounty saga and tomorrow we will be on a convict tour to learn about the penal colony!!""",
"""Managed an early start (7am) for a run and walk and then a swim in the Pacific at Emily Bay, which was really enjoyable. At 1pm we met up with a group and joined the 'Convict Settlement Tour'. The tour was very interesting but as we moved through the convict areas the information provided got more and more horrific. The treatment of the prisoners was really terrible and Norfolk Island finished up with the worst name in any of the penal colonies. After the tour the weather was so good we went for another swim. This was our last night on the Island so we went for a meal in an excellent restaurant. I had Red Emperor fish and Viv had a Prawn Salad!!""",
"""Had a walk around Emily Bay and packed for the return to Brisbane. Managed one more sight seeing trip, up Mount Pitt which is the highest hill on the Island at 1,000 feet. The whole Island can be seen from the top, a beautiful view. Caught the flight and it arrived in Brisbane 40 mins early. Back to Rick and Lynne's and a piece of barbecued Pork Belly, absolutely delicious.""",
"""Managed a run and bike ride. Then off into Brisbane for a short trip to the Gallery of Modern Art obviously with a stop for coffee etc. Had to get back to base to enjoy a seafood lunch on the house jetty at high tide. We had oysters, prawns and a home caught crab as well as a few glasses of the 'local' wine. This was the prelude to the NRL final between the Townsville Cowboys and the Brisbane Broncos, the first all Queensland final. Great match, and with 1 second to go the Cowboys produced a try to get level at 16 - 16. The goal attempt hit the inside of the upright and bounced out, so no score. Then the 'Golden Point' which the Cowboys got in 2 mins, so final score 17 - 16 to the Cowboys!""",
"""Quiet day today. A trip into Brisbane and visits to the Museum and the Maritime Museum. The Museum was not great, insufficient exhibits and not enough continuity but the Maritime was excellent with some real history. Temperature up to 35 deg!! Packing now for the trip to Tasmania. Leaving st 7.10 am tomorrow for the airport. """,
"""Arrived in Hobart mid afternoon (3 hrs from Brisbane) to a strong northerly wind which was maintaining a 31 deg! Walked around the port area which was interesting and had dinner at the hotel, which was excellent. Tomorrow we will be making an early start and driving to Port Arthur which was the Penal Settlement.""",
"""Set off early for a trip down the Tasman Peninsula. Beautiful country with rocky bays and causeways across inlets. Finally reached Port Stanley. This was the British Penal Colony set up with the first boat of convicts in 1830. The site is very large and although badly damaged by bush fires in 1877 and 1880 has been conserved for tourism. We had an introductory tour and did some detailed viewing in the museums. Very interesting. On the way back to Hobart we called in a coffee shop. Incredibly the proprietor was from Blacon near Chester and he was a Liverpool FC supporter and so he put a LFC top on my shoulders and me an Everton supporter!""",
"""The weather was glorious and after breakfast we joined the fast ferry (25 knots) to Mona, which was a 30 min run up the River Derwent. Mona is a modern art gallery built by a very rich man. The gallery is set up to challenge your senses and is buried underground in a huge chamber hollowed out of the rock on the edge of the river. The art is very modern and very difficult to grasp but that is the aim! After the return to Hobart we drove north east to a very english town called Richmond. The town is on the River Coal and has a stone bridge which is one of the oldest bridges in Australia and was built by convicts. Richmond is also a great area for vineyards and Tasmanian wines. Calling in a local wine store we got a private wine tasting and the wines are excellent if rather expensive.""",
"""Glorious day. After breakfast we looked at some of the craft and antique shops in Richmond. We then set off up the east coast through Swansea, Beaumaris and finally made St Helens. Are we at home? Some beautiful scenery, beaches and bays. Very quiet here, most of the restaurants close at 8pm! Finally found an Italian restaurant and had an excellent meal.""",
"""Travelled from St Helens over a mountain range to Derby. Derby was a massive tin mining area but had been devastated by a dam burst during an extremely heavy rainstorm in 1929. The town recovered and was in huge tin production during the Second World War. Since then it has finished with tin mining and is developing as a tourist attraction. The tin museum was very interesting and they have recently set up 12 mountain bike trails with a range of standards. We drove on to George Town where we visited the Bass and Flinders Museum. There was a full scale replica of the 'Norfolk' which Bass And Flinders sailed into the Tamar a River on a voyage to prove there was a Bass Strait between Australia and Tasmania.""",
"""Had a look at the Cataract in Launceston which is an impressive gorge on the South Esk River. Drove from Launceston to Devonport, where a large ferry runs to Melbourne. Devonport lies on the a River Mersey!! Another beautiful day and we moved on to Burnie. All these towns lie on the Bass Strait which is between Tasmania and Australia.""",
"""Left Burnie and travelled to Cradle Mountain. Most of the journey was through arable farm land but changed dramatically to rain forest as we climbed towards the mountain. The scenery changed again to moorland and finally we arrived in the valley to Cradle Mountain. The area is very popular with walkers, but we were just ahead of the summer season, so it was fairly quiet. Beautiful spot, the mountain is about 5,000 ft high and is shaped like a miners cradle. At the reception area there is a Tasmanian Devil and Quoll sanctuary which was interesting. Both animals are carnivores, the Devil tends to stay on the ground and the Quoll is tree climbing. We pressed on through a beautiful area to Strahan, which is a coastal town in the west facing the Southern Ocean. The sea comes straight in from the west, in fact straight from South America and 100 foot waves have been seen on this coast.""",
"""Left Strahan and travelled to Queenstown which was a Copper mining town and its heyday was between the 1880's and late 1960's. There were many mines in the area covering tin, zinc, gold and copper. The mining towns are old and run down but would have been boom towns at their peak and the area is completely stripped of vegetation although there are some signs of regeneration. There is also a strong possibility that the copper mine might be reopening shortly. The drive from here was spectacular, along winding roads, through thick rain forests and further on down the valleys with fruit tree orchards, through Derwent Bridge. At Derwent Bridge we saw an incredible exhibition of wood carvings by one artist, Greg Duncan. The building housing the sculptures is 100 metres long and houses 100 panels each one 3 metres high and 1 metre wide of laminated Huon Pine, each panel weighing 250 Kgs. Each panel is carved in representative pictures of life in the area. Huon Pine only grows in Tasmania and is greatly prized as it grows very slowly (0.008 ins diameter) per year and some of the trees are over 2,000 years old. The project is to be spread over 10 years and is breathtaking in its concept and execution. Finally we reached New Norfolk where we stayed the night.""",
"""After some exercise we left New Norfolk and travelled down on the west of Hobart to Mount Wellington, which is 4,000 ft high and sits right behind Hobart. The road up to the top is 6 miles long but the view over Hobart and the surrounding bays and inlets is really spectacular. Carried on down the South West along the coast, past apple, pear and then cherry orchards, through Margate, Kettering, Woodbridge, Cygnet and finally Huonville. On the way we stopped at an excellent museum which described the history of the area and how in 1967, a great bush fire destroyed most of the South West of Tasmania and how the area has slowly recovered.""",
"""Drove back from Huonville to Hobart via Franklin, Port Huon, Geeveston and Kimgston. A couple of interesting walks through the Huon forests and on to the airport and back to Brisbane. In the evening my we were invited to dinner at Rick and Lynne's friends, Herb and 'Snorter', a really good evening!""",
"""Some exercise to start the day and a leisurely day in Brisbane. A beautiful day, about 27 deg. Lunch at the Art Gallery and a stroll through the shops. In the evening we met some more friends of Rick and Lynne, Cam and Jenny. Initially we went to a pub in the West End of the city. This pub used to be an office where Rick worked in the 70's and was then a really wild area of the city but is now the hub of city life in the evenings. After a couple of beers we moved on to a Greek Restaurant and an excellent meal.""",
"""Went for a drive north of the city to a couple of art areas, Malaney and Montville. Both villages have a local art and craft background with lots of exhibitions and shops. The villages are fairly high up in the Blackall range overlooking the Sunshine Coast. We lunched in a restaurant overlooking the valleys down to the sea, very pleasant.""",
"""Exercised this morning and after breakfast we drove into Brisbane to look for some new biking clothes for Brian, which was ultimately successful. The weather was glorious again, about 24 deg, and Lynne had prepared a perfect picnic, which we took to another lovely spot on the Brisbane River, called Sherwood. Having eaten and drunk the picnic we played chess and read our books, very relaxing!! A stir fry for dinner and a plan for tomorrow to go to the Gold Coast!""",
"""Drove down the Gold Coast with Rick and Lynne, which was about 50 miles from base. Visited a number of beaches - Sufers Paradise, Kurrawa, Mermaid, Nobby, Miami, Burleigh Head, Palm Beach, Coolagatta and Kirra. All of these beaches were fabulous with reasonable surf, but not great conditions for surfing today. There were some fantastic houses with one actually right up to a surfing beach priced at 10 million AUD (5 million pounds)!! """,
"""Flew to Adelaide (2.5 hrs flight) to see David Ryan, Rebe, Daughter Isla and Tom their Son. David was a friend of Gareth's and we know the family extremely well. Weather was lovely as usual and we had a meal at their home along with David's Parents, Peter and Mary, who had arrived from the UK yesterday. We are here for 4 days and there are a number of excursions planned!""",
"""The early morning weather was disappointing with wind and some rain but it settled and after an early morning run and walk along the River Torres we breakfasted at the hotel. The hotel is located right in the middle of town so we headed to the Art Gallery. The Gallery is excellent with some Pre-Raphaelites, and William Morris furnishing materials and furniture. We enjoyed the Australian Art, many of the early artists having settled in Australia from Europe. On then to the Botanical Gardens which were huge and very well established with an avenue of Morton Bay Fig Trees which had been planted in 1850 and were just majestic. Also an indoor wet land area and a palm house. Met the Ryan family for dinner which was in a Tapas Bar. Tomorrow we are off to the famous Australian Vineyards!! """,
"""Set off this morning to visit the major wine growing areas around Adelaide. There are 74 vineyards that run wine tasting to show off their products. All the vineyards are very large with really imposing reception areas to handle the tasting. We visited 4 of the vineyards, Hugh Hamilton, Wirra Wirra, Rosemount and Samuels Gorge. We tried Granache, 'The Ratbag' Merlot, 'The Rascal' Shiraz, 'Scoundrel', 'O' over ice and Ruby 'O' all excellent, and well worth remembering for the future.""",
"""We all travelled to Mount Lofty, parked the car and walked up the track to the summit (800 metres), which was about 3,200 metres uphill all the way. Tremendous views from the summit over Adelaide and the sea. Came back a different route which was over 4,000 metres, all downhill. In the afternoon we took the tram to Glenelg, which is a seaside resort just round the corner from West Beach where the Ryan family live. A walk down the beach, a beer in the Surf Club and back to the Ryans for a barbecue. Fly back to Bisbane tomorrow.""",
"""Took early flight to Brisbane. Back to Rick and Lynne's for lunch and then watched the Gold Coast 600 km car race on TV. Street circuit with V8's racing up to speeds of 200 mph, very exciting compared to F1. Relaxed the rest of the day.""",
"""Went to watch Rick & Lynne's grandson Cooper, who is five, playing his Sunday morning game of football. In the afternoon we watched the second race in the Gold Coast 600km on TV. Only two days to go and Rick and Lynne have organised a farewell dinner this evening. They have invited Bruce & Rhonda, Herb & Snorter and Colin and Lorene. The dinner was a great success with Pork Belly from the barbecue with lots of vegetables.""",
"""Last day today, so we drove down to the ferry and boarded to travel to North Stradbroke Island. These islands form a barrier across Brisbane Bay and comprise North and South Stradbroke Island and Moreton Island, mainly sand and are about 20~30 miles long. The weather was glorious and a bus trip up to the North allowed us to walk around the top of the island with its gorges and beautiful beaches and surf. A bus back to the pub overlooking the sea was next with a fish basket and a glass of wine for lunch. Then back to to the ferry and home. As a final experience Rick had manufactured rotisseries each to hold a pork hock. The pictures show how they work.""",
"""A quick trip to a new Shopping Mall for last minute things and back home to a superb Stir Fry. Flight was taking off at 21.10. Packing bags was a problem. We finished up with an allowance of 74 Kgs and took 73.6 Kgs!! """
]

for i in range(0, len(diaries)):
    diary = diaries[i]
    print("start tagging diary #%s" % (i+1))
    diary_tags = tagger.tag_pos_doc(diary, True)
    # print(diary_tags)
    print("create pickle for tags of diary #%s" % (i+1))
    tagger.tags_to_pickle(diary_tags, "../diary_pickles/as_travel_" + str(i+1) + ".pkl")
    print()