  # code style
  
  def func(
    params_1,
    params_2
  ):
    '''
    what this func does?
    
    Parameters:
    param_name: param_type, is_optional
      explain details.
    
    Returns:
    
    '''
    
    # main code
    
    return param3
  
  # example from python package lifetimes 
  
  def _customer_lifetime_value(
    transaction_prediction_model,
    frequency,
    recency,
    T,
    monetary_value,
    time=12,
    discount_rate=0.01,
    freq="D"
):
    """
    Compute the average lifetime value for a group of one or more customers.

    This method computes the average lifetime value for a group of one or more customers.

    It also applies Discounted Cash Flow.

    Parameters
    ----------
    transaction_prediction_model:
        the model to predict future transactions
    frequency: array_like
        the frequency vector of customers' purchases (denoted x in literature).
    recency: array_like
        the recency vector of customers' purchases (denoted t_x in literature).
    T: array_like
        the vector of customers' age (time since first purchase)
    monetary_value: array_like
        the monetary value vector of customer's purchases (denoted m in literature).
    time: int, optional
        the lifetime expected for the user in months. Default: 12
    discount_rate: float, optional
        the monthly adjusted discount rate. Default: 1

    Returns
    -------
    :obj: Series
        series with customer ids as index and the estimated customer lifetime values as values
    """

    df = pd.DataFrame(index=frequency.index)
    df["clv"] = 0  # initialize the clv column to zeros

    steps = np.arange(1, time + 1)
    factor = {"W": 4.345, "M": 1.0, "D": 30, "H": 30 * 24}[freq]

    for i in steps * factor:
        # since the prediction of number of transactions is cumulative, we have to subtract off the previous periods
        expected_number_of_transactions = transaction_prediction_model.predict(
            i, frequency, recency, T
        ) - transaction_prediction_model.predict(i - factor, frequency, recency, T)
        # sum up the CLV estimates of all of the periods and apply discounted cash flow
        df["clv"] += (monetary_value * expected_number_of_transactions) / (1 + discount_rate) ** (i / factor)

    return df["clv"] # return as a series
