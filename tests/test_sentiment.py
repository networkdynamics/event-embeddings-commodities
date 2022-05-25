from scripts.commodity_prediction.commodity_sentiment import CommoditySentiment

def test_sentiment():
    txt = "market eye-sgx nifty flat; bharti infratel debuts. * nifty futures on the singapore exchange unchanged. the msci-asia pacific index, excluding japan gains 0.43 percent. * bharti infratel ltd debuts after raising about $760 million in india's biggest ipo in two years. traders expect shares to come under pressure due to concerns about the outlook for mobile tower operators. * foreign investors sold 1.33 billion rupees of stocks, while domestic institutions bought 1.33 billion rupees of stocks on thursday, when india's bse index fell 0.48 percent. * traders expect market to trade in a narrow range, with select oil marketing companies shares to be traced on talks of hike in diesel prices."

    com_sent = CommoditySentiment('crude_oil')
    sentiment = com_sent.get_sentiment(txt)


def main():
    test_sentiment()

if __name__ == '__main__':
    main()