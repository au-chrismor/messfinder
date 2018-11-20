import boto3
import os
import json

def respond(err, res = None):
    return {
        'statusCode': '400' if err else '200',
        'body': err.message if err else json.dumps(res),
        'headers': {
            'Content-Type': 'application/json',
        },
    }

def lambda_handler(event, context):
    try:
        ddb = boto3.resource('dynamodb')
        objectName = event['queryStringParameters']['objectName']
        table = ddb.Table(os.environ['OBJTABLE'])
        resp = table.get_item(
            Key = {
                'objectName': objectName
            }
        )
        ret = resp['Item']
    except:
        ret = {
            'isMess': 'true'
        }
    return respond(0, ret)
    